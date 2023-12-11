
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from circleguard import Circleguard, KeylessCircleguard, ReplayDir, User
from slider import Library

from notebooks.utils.replay_processing import get_embeddings, get_judgments, get_beatmap_context

EMBEDDING_DIM = 16
JUDGMENT_DIM = 4
BEATMAP_CONTEXT_DIM = 8


NAIVE_NOTES_PER_EXAMPLE = 8
SEQ_NOTES_PER_EXAMPLE = 64

MODELS = {
    "naive" : "./models/naive/naive_final.keras", 
    "seq" : "./models/seq/seq_final.keras", 
    "seq2seq" : "./models/seq2seq/seq2seq_final.keras"
}

CONTEXT_LABELS = [
    "ar",
    "circle_radius",
    "hp",
    "hitwindow_300",
    "hitwindow_100",
    "hitwindow_50",
    "hd",
    "dt",
]

EMBEDDING_LABELS = [
    "x_pos",
    "y_pos",
    "in_x_offset",
    "in_y_offset",
    "in_dist",
    "in_timedelta",
    "out_x_offset",
    "out_y_offset",
    "out_dist",
    "out_timedelta",
    "angle",
    "is_slider",
    "slider_duration",
    "slider_length",
    "slider_num_ticks",
    "slider_num_beats"
]

NULL_LABELS = [
    "p_300_null",
    "p_100_null",
    "p_50_null",
    "p_miss_null"
]

PROB_LABELS = [
    "p_300",
    "p_100",
    "p_50",
    "p_miss"
]

TRUE_LABELS = [
    "300",
    "100",
    "50",
    "miss"
]

# Naive Processing ==========================================================================================


def _naive_replay_to_np_features(replay, beatmap_library):
    
    beatmap_context = get_beatmap_context(replay, beatmap_library)
    beatmap_context = np.reshape(beatmap_context, (BEATMAP_CONTEXT_DIM,))
    
    embs = get_embeddings(replay, beatmap_library)
    replay_len = len(embs)
    
    if replay_len <= NAIVE_NOTES_PER_EXAMPLE:
        return None
    
    res = np.zeros((replay_len - NAIVE_NOTES_PER_EXAMPLE, NAIVE_NOTES_PER_EXAMPLE * EMBEDDING_DIM + BEATMAP_CONTEXT_DIM))
    
    for note_idx in range(0, replay_len - NAIVE_NOTES_PER_EXAMPLE):
        
        es = embs[replay_len - note_idx - NAIVE_NOTES_PER_EXAMPLE : replay_len - note_idx , :]
        es = np.reshape( es, (NAIVE_NOTES_PER_EXAMPLE * EMBEDDING_DIM,) )
    
        res[note_idx, : ] =  np.concatenate( (beatmap_context, es) )
        
    return res


def _naive_np_embs_to_notes_df(np_features, judgs, model):
    
    if np_features is None:
        return None
    
    probs = model.predict_on_batch(np_features)
    
    res_df = pd.DataFrame(
        np.concatenate(
            (np_features[:, :BEATMAP_CONTEXT_DIM], np_features[:, -EMBEDDING_DIM:], probs, judgs), 
            axis = 1
        ), columns = CONTEXT_LABELS + EMBEDDING_LABELS + PROB_LABELS + TRUE_LABELS
    )[::-1]
    
    return res_df.reset_index(drop = True)



# Seq Processing ==========================================================================================


def _seq_get_embedding_sequence(embeddings, replay_len, note_idx):
    res = None
    if replay_len - note_idx - SEQ_NOTES_PER_EXAMPLE >= 0:
        res = embeddings[ replay_len - note_idx - SEQ_NOTES_PER_EXAMPLE : replay_len - note_idx , :]
    else:
        res = embeddings[ 0 : replay_len - note_idx , :]  
    return res



def _seq_replay_to_np_features(replay, beatmap_library, cg):
    
    beatmap_context = get_beatmap_context(replay, beatmap_library)
    beatmap_context = np.reshape(beatmap_context, (BEATMAP_CONTEXT_DIM,))
    
    embs = get_embeddings(replay, beatmap_library)
    judgs = get_judgments(replay, cg)
    
    replay_len = len(embs)
    
    if replay_len <= SEQ_NOTES_PER_EXAMPLE:
        return None
    
    res = np.zeros((replay_len, SEQ_NOTES_PER_EXAMPLE, EMBEDDING_DIM + BEATMAP_CONTEXT_DIM))
    
    for note_idx in range(replay_len):
         
        embedding_sequence = _seq_get_embedding_sequence(embs,replay_len, note_idx)
        context_sequence = np.tile(beatmap_context, [ len(embedding_sequence), 1])
        
        sequence = np.concatenate([embedding_sequence, context_sequence], axis = 1)        
        padding = np.zeros(shape = [SEQ_NOTES_PER_EXAMPLE - len(embedding_sequence), EMBEDDING_DIM + BEATMAP_CONTEXT_DIM])
    
        res[note_idx, :, :] =  np.concatenate([padding, sequence], axis = 0)
        
    return res, judgs


def _seq_np_embs_to_notes_df(np_features, judgs, model):
    
    if np_features is None:
        return None
    
    probs = model.predict_on_batch(np_features)
    
    res_df = pd.DataFrame(
        np.concatenate(
            (np_features[:, -1, :], probs, judgs), 
            axis = 1
        ), columns = EMBEDDING_LABELS + CONTEXT_LABELS + PROB_LABELS + TRUE_LABELS
    )[::-1]
    
    return res_df.reset_index(drop = True)


# Seq2Seq Processing ======================================================================================


def _seq2seq_get_embedding_sequence(embeddings, judgments, replay_len, note_idx):
    
    embs = None
    judgs = None
    
    # hide last judgment
    if replay_len - note_idx - SEQ_NOTES_PER_EXAMPLE >= 0:
        embs = embeddings[ replay_len - note_idx - SEQ_NOTES_PER_EXAMPLE : replay_len - note_idx , : ]
        judgs = judgments[ replay_len - note_idx - SEQ_NOTES_PER_EXAMPLE : replay_len - note_idx - 1, : ]     
    else:
        embs = embeddings[ 0 : replay_len - note_idx , : ]  
        judgs = judgments[ 0 : replay_len - note_idx - 1 , : ]
    
    judgs = tf.concat([judgs, tf.zeros([1, JUDGMENT_DIM])], axis = 0)


    return tf.concat([embs, judgs], axis = 1)


def _seq2seq_replay_to_np_features(replay, beatmap_library, cg):
    
    beatmap_context = get_beatmap_context(replay, beatmap_library)
    beatmap_context = np.reshape(beatmap_context, (BEATMAP_CONTEXT_DIM,))
    
    embs = get_embeddings(replay, beatmap_library)
    judgs = get_judgments(replay, cg)
    
    replay_len = len(embs)
    
    if replay_len <= SEQ_NOTES_PER_EXAMPLE:
        return None
    
    res = np.zeros((replay_len, SEQ_NOTES_PER_EXAMPLE, EMBEDDING_DIM + JUDGMENT_DIM + BEATMAP_CONTEXT_DIM))
    
    for note_idx in range(replay_len):
         
        embedding_sequence = _seq2seq_get_embedding_sequence(embs, judgs, replay_len, note_idx)
        context_sequence = np.tile(beatmap_context, [len(embedding_sequence), 1])
        
        sequence = np.concatenate([embedding_sequence, context_sequence], axis = 1)        
        padding = np.zeros(shape = [SEQ_NOTES_PER_EXAMPLE - len(embedding_sequence), EMBEDDING_DIM + BEATMAP_CONTEXT_DIM + JUDGMENT_DIM])
    
        res[note_idx, :, :] =  np.concatenate([padding, sequence], axis = 0)
        
    return res, judgs


def _seq2seq_np_embs_to_notes_df(np_features, judgs, model):
    
    if np_features is None:
        return None
    
    probs = model.predict_on_batch(np_features)
    
    res_df = pd.DataFrame(
        np.concatenate(
            (np_features[:, -1, :], probs, judgs), 
            axis = 1
        ), columns = EMBEDDING_LABELS + NULL_LABELS + CONTEXT_LABELS + PROB_LABELS + TRUE_LABELS
    )[::-1]
    
    return res_df.reset_index(drop = True)


# General Processing ======================================================================================


def get_notes_df( cg, replay, model, model_type, beatmap_library ):
    
    if model_type == "naive":
        
        np_features =  _naive_replay_to_np_features( replay, beatmap_library)
        judgs = get_judgments(replay, cg)[NAIVE_NOTES_PER_EXAMPLE:]
    
        return _naive_np_embs_to_notes_df( np_features, judgs, model )
    
    if model_type == "seq":
        
        np_features, judgs = _seq_replay_to_np_features( replay, beatmap_library, cg )
        return _seq_np_embs_to_notes_df( np_features, judgs, model )
        
    if model_type == "seq2seq":
    
        np_features, judgs = _seq2seq_replay_to_np_features( replay, beatmap_library, cg )
        return _seq2seq_np_embs_to_notes_df( np_features, judgs, model )



def _likelihood_lambda(row):
    
    res = 1
    
    if row["300"]:
        res =  row["p_300"]
    elif row["100"]:
        res =  row["p_300"] + row["p_100"]
    elif row["50"]:
        res =  row["p_300"] + row["p_100"] + row["50"]
    
    return -np.log(res)


    
def compute_likelihood(notes_df, top_k = 256 ):
    likelihoods = notes_df.apply(_likelihood_lambda, axis = 1)
    return np.sum ( sorted(likelihoods, reverse = True)[:top_k] ) 



def init_parser():
   
    parser = argparse.ArgumentParser(prog = "osu490",
                                     description = "Difficulty estimator for osu490: an ML performance point algorithm.")

    parser.add_argument("--replay_dir", type = str, default = None, required = True,
                        help = "Path to directory of replays. Required.")
    parser.add_argument("--beatmap_dir", type = str, default = None, required = True,
                    help = "Path to beatmap directory. Beatmap names must be md5 hashes, like those in o!rdr replay dataset. Required.")  
    parser.add_argument("--model", type = str, default = "naive",
                        help = "Model type ('naive', 'seq', or 'seq2seq'). Defaults to 'naive'.")
    parser.add_argument("--output_name", type=str, default = None,
                        help = "Name of produced csv file (if not provided, result is printed to console)")
    parser.add_argument("--verbose", action = 'store_true',
                        help = "Adds print statements to the configuration process.")
    
    return parser


def configure_args(args):
    
    verbose = args.verbose
    
    if verbose:
        print("\nConfiguring Circleguard...")
        
    cg = KeylessCircleguard(
        slider_dir = args.beatmap_dir,
        cache = False
    )  
    
    if verbose:
        print("done")
        print("\nLoading replay_dir...")
        
    replay_dir = ReplayDir(args.replay_dir)
    cg.load(replay_dir)
    
    if verbose:
        print("done")
        print("\nLoading model...")
        
    model_type = "naive"
    if args.model in MODELS:
        model_type = args.model
    model = tf.keras.models.load_model(MODELS[model_type])
    
    if verbose:
        print("done")
        print("\nInitializing beatmap library...")
        
    beatmap_library = Library(args.beatmap_dir)
    
    if verbose:
        print(f"Num. of beatmaps in library: {len(beatmap_library.ids)}")
        print("done")
        print("\nComputing difficulty...")
    
    return cg, replay_dir, model, model_type, beatmap_library
    
    

if __name__ == "__main__":

    parser = init_parser()
    args = parser.parse_args()
       
    
    cg, replay_dir, model, model_type, beatmap_library = configure_args(args)
    
    res = []


    for idx, replay in enumerate(replay_dir):
        
        try:
            n_df = get_notes_df(cg, replay, model, model_type, beatmap_library) 
        except KeyError as e:
            print(f"\ncant find beatmap for one of {replay.username}'s replays\n")
            continue
        
        estimated_difficulty = compute_likelihood(n_df)
        
        username = replay.username
        beatmap = replay.beatmap(beatmap_library)
        beatmap_name = beatmap.display_name
        mod_str = str(replay.mods)
        nm_stars = 0
        
        count_300 = replay.count_300
        count_100 = replay.count_100
        count_50 = replay.count_50
        count_miss = replay.count_miss
        
        try:
            nm_stars = beatmap.stars()
        except:
            pass
        
        res.append([idx, username, beatmap_name, mod_str, nm_stars,
                    replay.count_300, replay.count_100, replay.count_50, replay.count_miss, estimated_difficulty])
        
    
    res = pd.DataFrame(res, columns = ["replay_idx", "username", "beatmap_name", "mod_str", "nm_stars",
                             "count_300", "count_100", "count_50", "count_miss", "estimated_difficulty"])
    
    if args.output_name:
        res.to_csv(f"{args.output_name}.csv", index = False)
    else:
        print(res)
        

    
    



