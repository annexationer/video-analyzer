import cv2

cap = cv2.VideoCapture('/Users/equalizer/Video/13.ts')
print(cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.isOpened())
# n = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#     print(f'{frame_pos=}, {n=}')
#     n += 1
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
#
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.isOpened())
# cap.release()
# cv2.destroyAllWindows()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)

cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

flag = False
for frame_index in range(total_frames - 1, total_frames // 2 - 1, -1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        if not flag:
            flag = True
        print(ret, frame_index, frame)
        cv2.imshow(f'frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        if flag:
            break

cap.release()
cv2.destroyAllWindows()
