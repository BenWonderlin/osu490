
import numpy as np

from circleguard.judgment import JudgmentType
from slider.beatmap import Circle, Slider


#############################################################
# ====================== Parse Notes ====================== #
#############################################################


def _compute_angle_cosine(prev_obj, curr_obj, next_obj):
    
    EPSLION = 1e-06
    cos = np.pi

    prev_vector = [ prev_obj.position.x - curr_obj.position.x, prev_obj.position.y - curr_obj.position.y ]
    next_vector = [ next_obj.position.x - curr_obj.position.x, next_obj.position.y - curr_obj.position.y ]

    prev_norm = (prev_vector[0] ** 2 + prev_vector[1] ** 2) ** 0.5
    next_norm = (next_vector[0] ** 2 + next_vector[1] ** 2) ** 0.5

    if prev_norm > EPSLION and next_norm > EPSLION:
        cos = (prev_vector[0] * next_vector[0] + prev_vector[1] * next_vector[1]) / (prev_norm * next_norm)

    return cos


    
def _get_hitobject_embedding(prev_obj, curr_obj, next_obj, mod_str):
     

    x_position = curr_obj.position.x
    y_position = curr_obj.position.y

    in_x_offset = 0.0
    in_y_offset = 0.0
    in_distance = 0.0
    in_timedelta = 5000.0 # in ms

    out_x_offset = 0.0
    out_y_offset = 0.0
    out_distance = 0.0
    out_timedelta = 5000.0

    cos_angle = -1

    is_slider = 0.0
    slider_duration = 0.0
    slider_length = 0.0
    slider_num_ticks = 0.0
    slider_num_beats = 0.0


    if prev_obj:
        in_x_offset = curr_obj.position.x - prev_obj.position.x
        in_y_offset = curr_obj.position.y - prev_obj.position.y
        in_distance = ( in_x_offset ** 2 + in_y_offset ** 2) ** 0.5
        in_timedelta = (curr_obj.time - prev_obj.time).microseconds / 1000

    if next_obj:
        out_x_offset = next_obj.position.x - curr_obj.position.x
        out_y_offset = next_obj.position.y - curr_obj.position.y
        out_distance = ( out_x_offset ** 2 + out_y_offset ** 2) ** 0.5
        out_timedelta = (next_obj.time - curr_obj.time).microseconds / 1000

    if prev_obj and next_obj:
        cos_angle = _compute_angle_cosine(prev_obj, curr_obj, next_obj)

    if type(curr_obj) == Slider:
        is_slider = 1.0
        slider_duration = (curr_obj.end_time - curr_obj.time).microseconds / 1000
        slider_length = curr_obj.length
        slider_num_ticks = curr_obj.ticks
        slider_num_beats = curr_obj.num_beats
        
    if "DT" in mod_str or "NC" in mod_str:
        in_timedelta *= 2.0/3
        out_timedelta *= 2.0/3
        slider_duration *= 2.0/3
    
    if "HR" in mod_str:
        y_position = 384 - y_position
        in_y_offset *= -1
        out_y_offset *= -1

    return np.array([
        x_position, y_position,
        in_x_offset, in_y_offset, in_distance, in_timedelta,
        out_x_offset, out_y_offset, out_distance, out_timedelta,
        cos_angle,
        is_slider, slider_duration, slider_length, slider_num_ticks, slider_num_beats
    ])



def _filter_out_spinners(objects):
    return [o for o in objects if type(o) in (Circle, Slider)]



def get_embeddings(replay, beatmap_library):

    beatmap = beatmap_library.lookup_by_md5(replay.beatmap_hash)
    hitobjects = _filter_out_spinners( beatmap.hit_objects() )
    mod_str = str(replay.mods)

    num_hitobjects = len(hitobjects)
    res = np.zeros((num_hitobjects, 16))

    res[0, :] = _get_hitobject_embedding( None, hitobjects[0], hitobjects[1], mod_str )

    for emb_idx in range(1, num_hitobjects - 1):
        res[emb_idx, :] = _get_hitobject_embedding( hitobjects[emb_idx - 1], hitobjects[emb_idx], hitobjects[emb_idx + 1], mod_str )

    res[num_hitobjects - 1 , :] = _get_hitobject_embedding( hitobjects[num_hitobjects - 2], hitobjects[num_hitobjects - 1], None, mod_str )

    return res




#############################################################
# ==================== Parse Judgments ==================== #
#############################################################


def _sort_judgments(judgments):
    return sorted(judgments, key = lambda j: j.hitobject.t)
    
    
def _encode_judgment(judgment):
     t = judgment.type
     return np.array([
          1 if t == JudgmentType.Hit300 else 0, 
          1 if t == JudgmentType.Hit100 else 0,
          1 if t == JudgmentType.Hit50 else 0,
          1 if t == JudgmentType.Miss else 0
     ], dtype = int)


def get_judgments(replay, cg):
    
     judgments = _sort_judgments( cg.judgments(replay) )
     res_len = len(judgments)
     res = np.zeros((res_len, 4), dtype = int)

     # could probably be written w/o for loop?
     for judgment_idx in range(res_len):
          res[judgment_idx, :] = _encode_judgment(judgments[judgment_idx])

     return res
 
 
 
 
#############################################################
# =================== Parse Beatmap Info ================== #
#############################################################


def _compute_ar(ar, mod_str):
    
    adj_ar = ar
    
    if "HR" in mod_str:
        adj_ar = min(1.4 * ar, 10)
    
    elif "EZ" in mod_str:
        adj_ar /= 2
        
    return adj_ar
    
    
def _convert_ar_to_approach_ms(ar, mod_str):
    
    ms = 1200
    
    if ar < 5:
        ms += 600 * (5 - ar) / 5
        
    elif ar > 5:
        ms -= 750 * (ar - 5) / 5
        
    if "DT" in mod_str or "NC" in mod_str:
        ms *= 2 / 3
    
    return ms


def _compute_cs(cs, mod_str):
    
    adj_cs = cs
    
    if "HR" in mod_str:
        adj_cs = min(1.3 * adj_cs, 10)
        
    elif "EZ" in mod_str:
        adj_cs /= 2
        
    return adj_cs


def _convert_cs_to_circle_radius(cs):
    return 54.4 - 4.48 * cs
    

def _compute_hp(hp, mod_str):
    
    adj_hp = hp
    
    if "HR" in mod_str:
        adj_hp = min(1.4 * adj_hp, 10)
        
    elif "EZ" in mod_str:
        adj_hp /= 2
        
    return adj_hp


def _compute_od(od, mod_str):
    
    adj_od = od
    
    if "HR" in mod_str:
        adj_od = min(1.4 * od, 10)
        
    elif "EZ" in mod_str:
        adj_od /= 2
    
    return adj_od
    

def _convert_od_to_hitwindows(od, mod_str):
    
    hitwindows = [
        80 - 6 * od,
        140 - 8 * od,
        200 - 10 * od
    ]
    
    if "DT" in mod_str or "NC" in mod_str:
        for idx in range(3):
            hitwindows[idx] *= 2 / 3
            
    return hitwindows
    

def get_beatmap_context(replay, beatmap_library):
    
    beatmap = beatmap_library.lookup_by_md5(replay.beatmap_hash)
    mod_str = str(replay.mods)
    
    ar = _compute_ar( beatmap.ar(), mod_str )
    cs = _compute_cs( beatmap.cs(), mod_str )
    hp = _compute_hp( beatmap.hp(), mod_str )
    od = _compute_od( beatmap.od(), mod_str )
    
    approach_ms = _convert_ar_to_approach_ms( ar, mod_str )
    circle_radius = _convert_cs_to_circle_radius(cs)
    hitwindow_300, hitwindow_100, hitwindow_50 = _convert_od_to_hitwindows(od, mod_str)

    return np.array([
        
        approach_ms,
        circle_radius,
        hp,
        
        hitwindow_300, # OD
        hitwindow_100,
        hitwindow_50,
        
        1.0 if "HD" in mod_str else 0.0,                        # visual mod, so need boolean
        1.0 if "DT" in mod_str or "NC" in mod_str else 0.0,    # apparently DT changes the animation (? idk)
        
        # mods not included: FL, SO, NF, PF, SO, V2, AT, RL, 
    
    ])
    
    
    
    
#############################################################
# ==================== Validate Replays =================== #
#############################################################


def _validate_replay_length(replay, beatmap_objects, replay_judgments):

    res = 0

    beatmap_objects_no_spinners = _filter_out_spinners(beatmap_objects)
    replay_objects = [ j.hitobject for j in replay_judgments ]

    if len(beatmap_objects_no_spinners) != len(replay_judgments):
        #print(f"Length mismatch between beatmap objects {(len(beatmap_objects_no_spinners))} and replay judgments {(len(replay_judgments))} of replay at {replay.path}.")
        res |= 1

    if len(beatmap_objects_no_spinners) != len(replay_objects):
        #print(f"Length mismatch between beatmap objects {(len(replay_objects))} and replay objects {(len(beatmap_objects))} of replay at {replay.path}.")
        res |= 2
    
    return res


def _validate_replay_hitcounts(replay, beatmap_objects, replay_judgments):

    judgment_encodings = [_encode_judgment(j) for j in replay_judgments]
    judgment_hitcounts = np.sum(judgment_encodings, axis = 0)
    replay_hitcounts = np.array([ replay.count_300, replay.count_100, replay.count_50, replay.count_miss])
    hitcount_err_arr = replay_hitcounts - judgment_hitcounts

    num_spinners = len(beatmap_objects) - len(_filter_out_spinners(beatmap_objects))

    if sum(hitcount_err_arr) != num_spinners:
        #print(f"Hitcount mismatch ({judgment_hitcounts} vs. {replay_hitcounts}, num_spinners = {num_spinners}) of replay at {replay.path}.")
        return 4
    
    return 0


def _validate_replay_objects(replay, beatmap_objects, replay_judgments):

    res = 0

    EPSILON = 1e-06
    beatmap_objects_no_spinners = _filter_out_spinners(beatmap_objects)
    replay_objects = [ j.hitobject for j in replay_judgments ]

    for idx, _ in enumerate(beatmap_objects_no_spinners):

        replay_obj = replay_objects[idx]
        beatmap_obj = beatmap_objects_no_spinners[idx]

        if abs( replay_obj.time - beatmap_obj.time.total_seconds() * 1000 ) >= EPSILON : 
            #print(f"Offset mismatch ({1.0 * replay_obj.time} vs. {beatmap_obj.time.total_seconds() * 1000}) at index {idx} of replay at {replay.path}.")
            res |= 8

        if "HR" not in str(replay.mods) and "EZ" not in str(replay.mods): 

            if replay_obj.x - beatmap_obj.position.x >= EPSILON:
                #print(f"Position mismatch (x={replay_obj.x} vs. x={beatmap_obj.position.x}) at index {idx} of replay at {replay.path}.")
                res |= 16
            
            if replay_obj.y - beatmap_obj.position.y >= EPSILON:
                #print(f"Position mismatch (y={replay_obj.y} vs. y={beatmap_obj.position.y}) at index {idx} of replay at {replay.path}.")
                res |= 32
    
    return res
            

def validate_replay( replay, beatmap, cg ):

    res = 0

    beatmap_objects = beatmap.hit_objects()
    replay_judgments = _sort_judgments( cg.judgments(replay) )

    res |= _validate_replay_length(replay, beatmap_objects, replay_judgments)
    res |= _validate_replay_hitcounts(replay, beatmap_objects, replay_judgments)
    res |= _validate_replay_objects(replay, beatmap_objects, replay_judgments)
            
    return res



#############################################################
# ===================== Filter Replays ==================== #
#############################################################


def filter_replay(replay, beatmap):

    res = 0
    
    mod_str = str(replay.mods)
    if "V2" in mod_str or "FL" in mod_str or "HT" in mod_str \
        or "RL" in mod_str or "AP" in mod_str or "AT" in mod_str:
        res |= 1
    
    if replay.count_miss > 128:
        res |= 2
    
    if beatmap.max_combo < 128 or beatmap.max_combo > 8192:
        res |= 4
    
    if beatmap.bpm_min() < 32 or beatmap.bpm_max() > 512:
        res |= 8
    
    return res