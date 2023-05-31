class Student:
    def __init__(self, student_id):
        self.student_id = student_id
        self.blink = 0
        self.count = 0
        self.emotions = {
            'Angry': 0,
            'Disgust': 0,
            'Fear': 0,
            'Happy': 0,
            'Sad': 0,
            'Surprise': 0,
            'Neutral': 0
        }

    def update_emotions(self, new_emotions):
        self.count += 1
        for emotion, value in new_emotions.items():
            self.emotions[emotion] += value

    def add_blink(self):
        self.blink += 1

    def display_info(self):
        print(f"Student ID: {self.student_id}")
        print(f"Blinks: {self.blink}")
        print("Emotions:")
        for emotion, value in get_emotions(self).items():
            print(f"{emotion}: {value}")
    def get_student_id(self):
        return self.student_id

    def get_blink(self):
        return self.blink

    def get_emotions(self):
        normalized_emotions = {}
        for emotion, value, in self.emotions.items():
            normalized_emotions[emotion] = value / self.count
        return normalized_emotions