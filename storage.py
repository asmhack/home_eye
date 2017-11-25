from scipy.spatial import distance
import numpy as np

class Storage(object):
    """
    Handles all faces vectors
    """

    def __init__(self):
        self.vectors = []

    def append(self, new_vector):
        self.vectors.append(new_vector)

    def extend(self, new_vectors, id):
        self.vectors.extend([[id, vec] for vec in new_vectors])
        # for vec in new_vectors:
        #     self.vectors.append([id, vec])

    def dump(self):
        pass

    def load(self):
        pass

    def match_vector(self, target_vector, id):
        from pprint import pprint
        matched_vectors = []
        for vector_dict in self.vectors:
            if vector_dict[0] != id:
                # print '-'*100
                # print self.vectors
                # print '#'*100
                # pprint(vector_dict[1])
                dist = distance.euclidean(target_vector, vector_dict[1])
                # print dist

                if dist <= 0.6:
                    matched_vectors.append([vector_dict[0], dist]) # id, distance

        if matched_vectors:
            return min(matched_vectors, key=lambda x: x[1]) # sort by distance

        return False
