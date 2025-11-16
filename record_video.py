import cv2

def record_video(output_file="output.mp4", fps=20.0, frame_width=640, frame_height=480):
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    print("Recording... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Write frame to the output file
        out.write(frame)

        # Show preview window
        cv2.imshow("Recording (press q to stop)", frame)

        # Stop when user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Recording saved to: {output_file}")

if __name__ == "__main__":
    record_video()
