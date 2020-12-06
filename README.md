# Adversarial-attack-on-mtcnn
Evaluate the AP of four different adversarial-patch pasting methods on two types of faces(no mask , mask on)

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
