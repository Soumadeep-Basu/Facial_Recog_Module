import cv2
import numpy as np 
import face_recognition



def find_faces_on_screen(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img)

    return face_locations


def get_encoded(): 
    user2 = face_recognition.load_image_file('Images_TestTrain/bTrain.jpg')
    user2 = cv2.cvtColor(user2, cv2.COLOR_BGR2RGB)
    user2 = cv2.rotate(user2, cv2.ROTATE_90_CLOCKWISE)


    user1 = face_recognition.load_image_file('Images_TestTrain/sTrain.jpg')
    user1 = cv2.cvtColor(user1, cv2.COLOR_BGR2RGB)
    user1 = cv2.rotate(user1, cv2.ROTATE_90_COUNTERCLOCKWISE)
   

   #for now we can keep it limited to two users

    user1 = face_recognition.face_encodings(user1)[0]
    user2 = face_recognition.face_encodings(user2)[0]


    li = [[user1,"User1"],[user2,"User2"]]

    return li

def main():

    TestImage = face_recognition.load_image_file('Images_TestTrain/bTest.jpg')
    TestImage = cv2.cvtColor(TestImage, cv2.COLOR_BGR2RGB)
    TestImage = cv2.rotate(TestImage, cv2.ROTATE_90_CLOCKWISE)
    
    # while(True):
    
    face_locations_on_screen = find_faces_on_screen(TestImage)

    # user_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    user_image_encode = face_recognition.face_encodings(TestImage)

    for i in face_locations_on_screen:
        cv2.rectangle(TestImage, (i[3], i[0]), (i[1], i[2]), (255,0,255), 2)
        
    encoded_list = get_encoded()


    found = 0
    for i in user_image_encode:
        if found == 1:
            break
        for j in encoded_list:
            val = face_recognition.compare_faces([i], j[0] )
            if(val[0] == True ):
                print("User Identified as :", j[1])
                found = 1
                break

    

        


    cv2.imshow("UserWindow",TestImage)
    cv2.waitKey(0)




if __name__=='__main__':
    main()
