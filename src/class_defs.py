import numpy as np
import spacy

print("Loading spaCy model...", end=" ")
nlp = spacy.load("en_core_web_md")
print("Done")

class RecipeAction:
    def __init__(self, ID, position, text, label):

        self.recipeID = ID      # ID of the recipe
        self.pos = position     # Position in the recipe instructions    
        self.text = text        # Text of the recipe instruction
        self.label = label          # One-word action label of the instruction
        self.label_embedding = self.generate_embedding(label)  # Embedding of the action label
        self.text_embedding = self.generate_embedding(text)  # Embedding of the instruction text
        self.state = None       # Defined state of the action from cluster

    def generate_embedding(self, text):
        """Generate and return the embedding for the given text."""
        return nlp(text).vector
    
    def set_label(self, label):
        """Generate and return the action label for the given text."""
        self.label = label

    def normalize_position(self, length):
        """Normalize the position of the action based on the length of the recipe."""
        self.pos = self.pos / length

    def embedding(self):
        """Return the embedding of the recipe step."""
        return self.embedding

    def set_state(self, state):
        """Set the state of the recipe step."""
        print(f"Setting state of action {self.label} to {state}")
        self.state = state

    def update_text(self, new_text, new_label):
        """Update the text of the recipe step and regenerate its embedding."""
        self.text = new_text
        self.embedding = self.generate_embedding(new_text)
        self.label = new_label

    def from_string(self, string):
        """Create a RecipeAction object from a string."""
        ID, pos, text, label = string.split("|")
        print(f"ID: {ID}, pos: {pos}, text: {text}, label: {label}")
        input()
        return RecipeAction(int(ID), int(pos), text, label)

    def __str__(self):
        """String representation of the recipe step."""
        return f"Recipe ID: {self.recipeID} | Position: {self.pos}, Label: {self.label}"

    def __eq__(self, other):
        """Equality check based on recipe ID and position."""
        return self.recipeID == other.recipeID and self.pos == other.pos

    def __lt__(self, other):
        """Less than comparison based on position in the recipe."""
        return self.pos < other.pos

class StateCluster:
    def __init__(self, ID, label, centroid):
        #print(f"Creating cluster {ID} with label '{label}'")
        self.clusterID = ID         # ID of the cluster
        self.position = None         # Position derived from the average position of actions in the cluster
        self.label = label          # Label of the cluster
        self.centroid = centroid    # Centroid of the cluster
        self.actions = []      # List of RecipeAction objects in the cluster

    def add_action(self, action):
        """Add an action to the cluster."""
        print(f"Adding action {action.label} to cluster {self.clusterID}")
        assert type(action) == RecipeAction, "Input must be a RecipeAction object"
        self.actions.append(action)
        self.position = np.mean([action.pos for action in self.actions])

    def remove_action(self, action):
        """Remove an action from the cluster."""
        self.actions.remove(action)

    def position(self):
        """Return the position of the cluster."""
        return self.position.__format__(".2f")

    def actions(self):
        """Return the list of actions in the cluster."""
        return self.actions
    
    def set_label(self, label):
        """Set the label of the cluster."""
        self.label = label

    def update_centroid(self):
        if self.actions:
            self.centroid = np.mean([action.embedding for action in self.actions], axis=0)
        else:
            self.centroid = np.zeros_like(self.centroid)

    def __str__(self):
        """String representation of the cluster."""
        return f"Cluster ID: {self.clusterID:>{3}} | Label: {self.label:>{15}} | Position: {self.position.__format__(".2f")} | # of Actions: {len(self.actions):>{1}}"