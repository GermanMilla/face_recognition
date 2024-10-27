from pathlib import Path

import face_recognition
import pickle
from collections import Counter
import pytesseract
from PIL import Image


DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")


#instalar tesseract desde https://tesseract-ocr.github.io/tessdoc/Installation.html
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


def id_validation(filepath):
    return "Documento Unico de Identidad" in pytesseract.image_to_string(Image.open(filepath))

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image.astype('uint8'), model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)


        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

# entrenar el modelo
# encode_known_faces()

def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )
    
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    
    #leftmost_face_index = min(range(len(input_face_locations)), key=lambda i: input_face_locations[i][3])
    #leftmost_face_location = input_face_locations[leftmost_face_index]
    #leftmost_face_encoding = input_face_encodings[leftmost_face_index]
    
    results = []

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if name:
            results.append({name: bounding_box})
    
    return results
        


def _recognize_face(unknown_encoding, loaded_encodings, tolerance=0.50):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], 
        unknown_encoding,
        tolerance=tolerance
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

#test call
test = recognize_faces("Dui.png")

print(id_validation("Dui.png"))
print(test)

