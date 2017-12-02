from scipy.spatial import distance


class Storage(object):
    """
    Handles all faces vectors
    """

    def __init__(self, model):
        self.vectors = model
        self.current_face_id = 0

        if model:
            max_row = max(model, key=lambda x: x[0])
            if max_row:
                self.current_face_id = max_row[0] + 1

    def append(self, new_vector):
        self.vectors.append(new_vector)

    def extend(self, new_vectors, id):
        self.vectors.extend([[id, vec] for vec in new_vectors])

    def get_total_images_for_label(self, label):
        return len(filter(lambda x: x[0]==label, self.vectors))

    def dump(self):
        pass

    def load(self):
        pass

    def match_vector(self, target_vector, id):
        matched_vectors = []
        for vector_dict in self.vectors:
            if vector_dict[0] != id:
                dist = distance.euclidean(target_vector, vector_dict[1])
                if dist <= 0.6:
                    matched_vectors.append([vector_dict[0], dist])  # id, distance

        if matched_vectors:
            return min(matched_vectors, key=lambda x: x[1])  # sort by distance

        return False
