# Adversarial-attack-on-mtcnn
Evaluate the AP of four different adversarial-patch pasting methods on two types of faces(no mask , mask on)

### Real world attack demo
Video : <https://www.youtube.com/watch?v=mJUsIuD5B3Q&feature=youtu.be&ab_channel=%E6%B4%AA%E5%95%86%E7%A8%8B>>

### Patch pasting
The pasting coordinates are in patch_coordinate.txt file
  
- chin
- forehead
- frame 
- mouth

### Picture
- The test samples have two types of faces
- Each type has five different sizes of faces(standing one to five meter away from the camera)

### Requirements
- install mtcnn

  `pip install mtcnn`

- change mtcnn's factory.py
    
    `keras.layers -> tensorflow.keras.layers`
    
    `keras.models -> tensorflow.keras.models`
    
- save mtcnn.py's scales into scales.npy
    
    `np.save('scales', scales)`
