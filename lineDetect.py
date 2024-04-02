import cv2
import numpy as np
from midline import midCalc
count = 0
arrow_countdown = 0
rightturn = cv2.imread('rightarrow.png', cv2.IMREAD_UNCHANGED)
roverlay_height, roverlay_width, _ = rightturn.shape

leftturn = cv2.imread('leftarrow.png', cv2.IMREAD_UNCHANGED)
loverlay_height, loverlay_width, _ = leftturn.shape

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_thickness = 5
text_color = (0, 255, 0)

def lineDetection(cap):
    global count, arrow_countdown
    ret, frame = cap.read()

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #matchtemplate for right arrow

    right_arrow_template = cv2.imread('right.png')
    right_gray = cv2.cvtColor(right_arrow_template, cv2.COLOR_BGR2GRAY)
    resultr = cv2.matchTemplate(gray, right_gray, cv2.TM_CCOEFF_NORMED)

    #matchtemplate for left arrow

    left_arrow_template = cv2.imread('left.png')
    left_gray = cv2.cvtColor(left_arrow_template, cv2.COLOR_BGR2GRAY)
    resultl = cv2.matchTemplate(gray, left_gray, cv2.TM_CCOEFF_NORMED)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    # Find Canny edges
    edged = cv2.Canny(blurred, 30, 200)

    locr = cv2.findNonZero((resultr >= 0.55).astype(int))

    locl = cv2.findNonZero((resultl >= 0.55).astype(int))

    # Perspective transform
    pts1 = np.float32([[570, 490], [740, 490],
                       [420, 700], [1100, 700]])

    pts2 = np.float32([[570, 0], [740, 0],
                       [500, 700], [1100, 700]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(edged, matrix, (1280, 720))

    height, width = result.shape

    # Only focus on the lower half of the screen
    polygon = np.array([[
        (int(width * 0.25), int(height * 0.92)),  # Bottom-left point
        (int(width * 0.42), int(height * 0.3)),  # Top-left point
        (int(width * 0.60), int(height * 0.3)),  # Top-right point
        (int(width * 0.80), int(height * 0.92)),  # Bottom-right point
    ]], np.int32)

    mask = np.zeros_like(result)

    cv2.fillPoly(mask, polygon, 255)

    roi = cv2.bitwise_and(result, mask)

    # Initialize dictionaries to store lines based on slope
    positive = []
    negative = []

    # Finding Lines
    lines = cv2.HoughLinesP(
        roi,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=30,  # Min number of votes for a valid line
        minLineLength=50,  # Min allowed length of a line
        maxLineGap=999  # Max allowed gap between line segments for joining them
    )

    slopespx1 = []
    slopespy1 = []
    slopespx2 = []
    slopespy2 = []
    slopesnx1 = []
    slopesnx2 = []
    slopesny1 = []
    slopesny2 = []



    # Iterate over detected lines
    if lines is not None:
        for line in lines:
            # Extract line endpoints
            x1, y1, x2, y2 = line[0]

            # Reverse perspective transform on endpoints
            pts = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32)
            reverse_transformed_pts = cv2.perspectiveTransform(pts, np.linalg.inv(matrix))

            # Coordinates after reverse transformation
            x1_r, y1_r = reverse_transformed_pts[0][0]
            x2_r, y2_r = reverse_transformed_pts[1][0]

            slope = (y2_r-y1_r)/(x2_r-x1_r)

            if -0.1 <= slope <= 0.1:
                break

            if slope > 0: #get all positive or negative slopes
                slopespx1.append(x1_r)
                slopespy1.append(y1_r)
                slopespx2.append(x2_r)
                slopespy2.append(y2_r)
            elif slope < 0:
                slopesnx1.append(x1_r)
                slopesny1.append(y1_r)
                slopesnx2.append(x2_r)
                slopesny2.append(y2_r)
            else:
                pass

        try:
            #avg all positive and negative slopes
            pavgx1 = sum(slopespx1)/len(slopespx1)
            pavgx2 = sum(slopespx2)/len(slopespx2)
            pavgy1 = sum(slopespy1) / len(slopespy1)
            pavgy2 = sum(slopespy2) / len(slopespy2)




            navgx1 = sum(slopesnx1) / len(slopesnx1)
            navgx2 = sum(slopesnx2) / len(slopesnx2)
            navgy1 = sum(slopesny1) / len(slopesny1)
            navgy2 = sum(slopesny2) / len(slopesny2)



            roi_height = int(520)  # Assuming the ROI height is 10% of the frame height

            # Extend positive slope line

            if len(slopespx1) > 0 and len(slopespx2) > 0 and len(slopespy1) > 0 and len(slopespy2) > 0:
                positive_slope = (pavgy2 - pavgy1) / (pavgx2 - pavgx1)
                x2_extp = int(pavgx2 + (roi_height - pavgy2) / positive_slope)
                y2_extp = roi_height

            # Extend negative slope line
            if len(slopesnx1) > 0 and len(slopesnx2) > 0 and len(slopesny1) > 0 and len(slopesny2) > 0:
                negative_slope = (navgy2 - navgy1) / (navgx2 - navgx1)
                x2_extn = int(navgx2 + (roi_height - navgy2) / negative_slope)
                y2_extn = roi_height

            frame_height = frame.shape[0]

            # Extend positive slope line to the bottom of the frame
            if len(slopespx1) > 0 and len(slopespx2) > 0 and len(slopespy1) > 0 and len(slopespy2) > 0:
                positive_slope = (pavgy2 - pavgy1) / (pavgx2 - pavgx1)
                x2_ext_bottomp = int(pavgx2 + (frame_height - pavgy2) / positive_slope)
                y2_ext_bottomp = frame_height


            # Extend negative slope line to the bottom of the frame
            if len(slopesnx1) > 0 and len(slopesnx2) > 0 and len(slopesny1) > 0 and len(slopesny2) > 0:
                negative_slope = (navgy2 - navgy1) / (navgx2 - navgx1)
                x2_ext_bottomn = int(navgx2 + (frame_height - navgy2) / negative_slope)
                y2_ext_bottomn = frame_height


            #draw positive line
            cv2.line(frame, (int(x2_ext_bottomp), int(y2_ext_bottomp)), (int(x2_extp), int(y2_extp)), (0, 255, 0), 8)

            #draw negative line
            cv2.line(frame, (int(x2_ext_bottomn), int(y2_ext_bottomn)), (int(x2_extn), int(y2_extn)), (0, 255, 0), 8)


            #find midline
            x1, y1, x2, y2 = midCalc(x2_ext_bottomp, x2_extp, y2_ext_bottomp, y2_extp, x2_ext_bottomn, x2_extn, y2_ext_bottomn, y2_extn)


            cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 8)



        except ZeroDivisionError:
            pass


    if locr is not None and locl is not None:
        try:
            if resultr > resultl:
                if locr is not None:
                    count = 0
                    print('turn right')
                    overlay_image = rightturn[:, :, :3]  # Extract RGB channels without alpha
                    overlay_mask = rightturn[:, :, 3]  # Extract alpha channel

                    # Get the region where the overlay will be placed
                    roi = frame[50:50 + roverlay_height, 460:460 + roverlay_width]

                    # Apply the overlay using the alpha mask
                    for c in range(3):
                        roi[:, :, c] = np.where(overlay_mask[:, :] == 255,
                                                overlay_image[:, :, c],
                                                roi[:, :, c])

                    # Update the frame with the overlay
                    frame[50:50 + roverlay_height, 460:460 + roverlay_width] = roi

                elif count <= 75:
                    count+=1
                    print('turn right')
                    overlay_image = rightturn[:, :, :3]  # Extract RGB channels without alpha
                    overlay_mask = rightturn[:, :, 3]  # Extract alpha channel

                    # Get the region where the overlay will be placed
                    roi = frame[50:50 + roverlay_height, 460:460 + roverlay_width]

                    # Apply the overlay using the alpha mask
                    for c in range(3):
                        roi[:, :, c] = np.where(overlay_mask[:, :] == 255,
                                                overlay_image[:, :, c],
                                                roi[:, :, c])

                    # Update the frame with the overlay
                    frame[50:50 + roverlay_height, 460:460 + roverlay_width] = roi

                else:
                    pass

            else:
                if locl is not None:
                    count = 0
                    print('turn left')
                    overlay_image = leftturn[:, :, :3]  # Extract RGB channels without alpha
                    overlay_mask = leftturn[:, :, 3]  # Extract alpha channel

                    # Get the region where the overlay will be placed
                    roi = frame[50:50 + loverlay_height, 460:460 + loverlay_width]

                    # Apply the overlay using the alpha mask
                    for c in range(3):
                        roi[:, :, c] = np.where(overlay_mask[:, :] == 255,
                                                overlay_image[:, :, c],
                                                roi[:, :, c])

                    # Update the frame with the overlay
                    frame[50:50 + loverlay_height, 460:460 + loverlay_width] = roi

                elif count <= 75:
                    count += 1
                    print('turn left')
                    overlay_image = leftturn[:, :, :3]  # Extract RGB channels without alpha
                    overlay_mask = leftturn[:, :, 3]  # Extract alpha channel

                    # Get the region where the overlay will be placed
                    roi = frame[50:50 + loverlay_height, 460:460 + loverlay_width]

                    # Apply the overlay using the alpha mask
                    for c in range(3):
                        roi[:, :, c] = np.where(overlay_mask[:, :] == 255,
                                                overlay_image[:, :, c],
                                                roi[:, :, c])

                    # Update the frame with the overlay
                    frame[50:50 + loverlay_height, 460:460 + loverlay_width] = roi
        except ValueError:
            pass

    elif locr is not None:

        count = 0
        print('turn right')
        overlay_image = rightturn[:, :, :3]  # Extract RGB channels without alpha
        overlay_mask = rightturn[:, :, 3]  # Extract alpha channel

        # Get the region where the overlay will be placed
        roi = frame[50:50 + roverlay_height, 460:460 + roverlay_width]

        # Apply the overlay using the alpha mask
        for c in range(3):
            roi[:, :, c] = np.where(overlay_mask[:, :] == 255,
                                    overlay_image[:, :, c],
                                    roi[:, :, c])

        # Update the frame with the overlay
        frame[50:50 + roverlay_height, 460:460 + roverlay_width] = roi

    elif locl is not None:
        count = 0
        print('turn left')
        overlay_image = leftturn[:, :, :3]  # Extract RGB channels without alpha
        overlay_mask = leftturn[:, :, 3]  # Extract alpha channel

        # Get the region where the overlay will be placed
        roi = frame[50:50 + loverlay_height, 460:460 + loverlay_width]

        # Apply the overlay using the alpha mask
        for c in range(3):
            roi[:, :, c] = np.where(overlay_mask[:, :] == 255,
                                    overlay_image[:, :, c],
                                    roi[:, :, c])

        # Update the frame with the overlay
        frame[50:50 + loverlay_height, 460:460 + loverlay_width] = roi


    else:
        pass



    # cv2.imshow('frame', frame)
    # cv2.imshow('warp', result)
    # cv2.imshow('mask', mask)
    # cv2.imshow('roi', roi)


    return frame


    # # Display frames
