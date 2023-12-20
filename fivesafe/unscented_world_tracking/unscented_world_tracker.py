import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def abs_dist_batch(detections, trackers):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  trackers = trackers[:,0:2]
  
  dist_array = cdist(detections, trackers, "euclidean")
                                           
  return(dist_array)  


class UnscentedKalmanWorldTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox, dt):
    """
    Initialises a tracker using initial bounding box.
    """
    self.dt = dt
    #define constant velocity model
    self.kf = UnscentedKalmanFilter(dim_x=4, dim_z=2) 
    # [x, y, vx, vy]
    self.kf.F = np.array([[1,0, self.dt, 0],
                          [0,1,0, self.dt],
                          [0,0,1,0],
                          [0,0,0,1]])
    
    self.kf.H = np.array([[1,0,0,0],
                          [0,1,0,0]])
    R_std = 10
    Q_std = .3
    self.kf.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=self.dt, var=Q_std**2)
    self.kf.Q = block_diag(q, q)
    self.kf.alpha = 1.2

    self.kf.x = np.array([[bbox[0], bbox[1], 0, 0]]).T
    self.time_since_update = 0
    self.id = UnscentedKalmanWorldTracker.count
    UnscentedKalmanWorldTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

    #self.original_id = bbox[5] # <--- add to keep track of original IDs

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    #self.original_id = bbox[5]  # <--- update to keep track of original IDs
    self.kf.update((bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x[0:2]


def associate_detections_to_trackers(detections,trackers, dist_threshold = 5*81/5):
  """
  Assigns detections to tracked object (both represented as World Positions)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,4),dtype=int)

  dist_matrix = abs_dist_batch(detections, trackers)

  if min(dist_matrix.shape) > 0:
    a = (dist_matrix < dist_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(dist_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low Distance
  matches = []
  for m in matched_indices:
    if(dist_matrix[m[0], m[1]]>dist_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class UnscentedWorldSort(object):
  def __init__(self, max_age=5, min_hits=0, dist_threshold = 5*81/5, dt_std = 1/30): # initial weights: self, max_age=1, min_hits=3, iou_threshold=0.3
    """
    Sets key parameters for SORT
    """
    self.dt_std = dt_std
    self.max_age = max_age
    self.min_hits = min_hits
    self.dist_threshhold = dist_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 2))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 4))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()
      trk[:] = [pos[0][0], pos[1][0], pos[2][0], pos[3][0]]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.dist_threshhold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = UnscentedKalmanWorldTracker(dets[i,:], self.dt_std)
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()
        if (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  #(trk.time_since_update < 1) and
          ret.append(np.concatenate((d[0], d[1], [trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive  # <--- add [trk.original_id] to the returned set
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,4))