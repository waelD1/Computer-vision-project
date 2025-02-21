
from sklearn.cluster import KMeans


class TeamAssigner:

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}


    def get_clustering_model(self, image):
        # Reshape the image into 2D array
        image_2d = image.reshape(-1, 3)

        # Fit the KMeans model
        # We choose option k-means++ to converge faster
        kmeans = KMeans(n_clusters=2, init = "k-means++", n_init=1, random_state=42)
        kmeans.fit(image_2d)

        return kmeans


    def get_player_color(self, frame, bbox):
        """
            Extracts the dominant color of a player's jersey from a given frame using K-Means clustering.

            Parameters:
            -----------
            frame : numpy.ndarray
                The image frame from which the player's color needs to be extracted.
            bbox : tuple
                A bounding box (x_min, y_min, x_max, y_max) that defines the player's region in the frame.

            Returns:
            --------
            numpy.ndarray
                The RGB color of the player's jersey as a NumPy array.

            Description:
            ------------
            1. Crops the player's region from the frame using the bounding box.
            2. Extracts the top half of the cropped image, assuming the jersey is more visible in this region.
            3. Applies K-Means clustering to segment the image into color clusters.
            4. Determines the player's jersey color by identifying the least common cluster in the image corners.
            5. Returns the centroid color of the identified cluster.
        """

         # crop the player image from the frame
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        top_half_image = image[0 : int(image.shape[0] / 2), :]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels into the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key = corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color





    def assign_team_color(self, frame, player_detections):
        
        player_colors = []
        for track_id, player_detection in player_detections.items():
            bbox =  player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init = "k-means++", n_init = 1, random_state=42)
        kmeans.fit(player_colors)

        # Save the kmeans model to use it later
        self.kmeans = kmeans

        # We assign the color for each team based on the cluster centers
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    
    def get_player_team(self, frame, player_bbox, player_id):

        # First we check the player ID
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team_id according to the player color
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        # We want the team_id to be 1 or 2
        team_id +=1

        # Save the player_id and team_id to not have to run the kmeans model again for this player
        self.player_team_dict[player_id] = team_id

        return team_id


