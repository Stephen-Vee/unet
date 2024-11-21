import cv2
def show_pic(pic, name="unnamed", type = 0, key = 'q'):
    if pic is None:
        print("显示图片为空")
    cv2.imshow(name, pic)
    cur_key = cv2.waitKey(0)
    if type == 0:
        cv2.destroyAllWindows()
    else :
        if cur_key == ord(key):
            cv2.destroyAllWindows()