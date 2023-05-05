def face_detection():
    import cv2
    import dlib
    import numpy as np
    import face_recognition

    # Load the input image
    path_image = input("Please enter your image path: ")
    input_image = cv2.imread(path_image)

    # Load the face detector and the facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Convert the input image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # If no faces are detected, rotate the image by 180 degrees and try again
    if len(faces) == 0:
        rotated_image = cv2.rotate(input_image, cv2.ROTATE_180)
        rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        faces = detector(rotated_gray)
        
        # If still no faces are detected, exit the program
        if len(faces) == 0:
            print('No faces detected.')
            exit()

        # Process the first detected face in the rotated image
        face = faces[0]
        
        # Get the facial landmarks for the face
        landmarks = predictor(rotated_gray, face)
        
        # Extract the coordinates of the two eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        
        # Compute the slope of the line that connects the center of the two eyes
        slope = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
        
        # Rotate the output image by 180 degrees if the face is upside down
        if abs(slope) > 1.0:
            output_image = cv2.rotate(rotated_image, cv2.ROTATE_180)
        else:
            output_image = rotated_image
        
    else:
        # Process the first detected face in the input image
        face = faces[0]
        
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)
        
        # Extract the coordinates of the two eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        
        # Compute the slope of the line that connects the center of the two eyes
        slope = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
        
        # Rotate the output image by 180 degrees if the face is upside down
        if abs(slope) > 1.0:
            output_image = cv2.rotate(input_image, cv2.ROTATE_180)
        else:
            output_image = input_image

    # Display the output image
    cv2.imshow('Output Image', output_image)

    # Wait for a key press to exit
    cv2.waitKey(0)

    # Release the resources
    cv2.destroyAllWindows()

    target_face_encoding = face_recognition.face_encodings(output_image)[0]

    # Load the uploaded video
    path_video = input("Please enter your video path: ")
    cap = cv2.VideoCapture(path_video)

    # Process the frames of the video
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            # End of the video
            break
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Compute the face encodings for all the faces in the frame
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Check if the target face is present in the frame
        for face_encoding, face_location in zip(face_encodings, face_locations):
            distance = np.linalg.norm(face_encoding - target_face_encoding)
            if distance < 0.6:
                # The target face is present in the frame
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Output a message or a notification
                print('Target face detected in the video')
            else:
                print('Target face not detected in the video')
        # Display the output frame
        cv2.imshow('Output Frame', frame)
        
        # Wait for a key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()

face_detection()