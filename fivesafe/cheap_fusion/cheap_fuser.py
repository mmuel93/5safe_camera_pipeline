import numpy as np
from scipy.spatial import KDTree
import networkx as nx
from networkx.algorithms.components.connected import connected_components

class CheapFusion():
    def __init__(self, dist_threshhold, scalefactor):
        self.dist_threshhold = dist_threshhold * scalefactor

    def fuse(self, tracks):
        """
        This Fuction fuses points that are too close together.
        Input: self (distance threshhold in meters)
                tracks(np.array of all tracks from the World Tracker (Positions are given in Pixelcoordinates))
        """
        fused_track_array = np.empty((0, 6))
        if len(tracks) < 2:
            return tracks
        trk_assignment_arr = tracks[:,0:2]
        tree = KDTree(trk_assignment_arr)
        rows_to_fuse = tree.query_pairs(r=self.dist_threshhold)

        G = nx.Graph()
        G.add_edges_from(rows_to_fuse)
        components = list(nx.connected_components(G))

        flat_list = [x for xs in components for x in xs]

        for i in range(len(trk_assignment_arr)):
            if i not in flat_list:
                components.append({i})
        components = list(components)
        # Fuse Candidates are compared. Oldest wins and appended to return array. If Two components have same age, first Component wins
        for component in components:
            component = list(component)
            if len(component) < 2:
                fused_track_array = np.append(fused_track_array, np.array([tracks[component[0]]]), axis=0)
            else:
                curr_winner_idx = component[0]
                curr_winner_age = tracks[curr_winner_idx][5]
                for idx in component:
                    if tracks[idx][5] > curr_winner_age:
                        curr_winner_idx = idx
                        curr_winner_age = tracks[curr_winner_idx][5]
                fused_track_array = np.append(fused_track_array, np.array([tracks[curr_winner_idx]]), axis=0)
        print("Before Fusion:")
        print(tracks)
        print("--------------------------------------------------------------------------")
        print("After Fusion:")
        print(fused_track_array)
        if tracks.shape != fused_track_array.shape:
            print("Fusion successfull")
        return fused_track_array
    