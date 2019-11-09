import cv2
import ImageProcUtils
import numpy as np

lena_image = cv2.imread('/home/rajan/DIS/HW1/lena.png', -1)
#lena_image = cv2.imread('/home/rajan/DIS/HW1/wolves.png', -1)
lena_image_gray=cv2.cvtColor(lena_image,cv2.COLOR_BGR2GRAY)

input_option=(lena_image_gray, lena_image)

kernel_cluster=(                 #All the filters are defined with 3 channels for convenience with RGB images

np.array([
   [[1/9, 1/9, 1/9],             #Smootherning filer
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]],

    [[1/9, 1/9, 1/9],
     [1/9, 1/9, 1/9],
     [1/9, 1/9, 1/9]
    ],

    [[1/9, 1/9, 1/9],
     [1/9, 1/9, 1/9],
     [1/9, 1/9, 1/9]
    ]
]),

np.array([                       #Prewitt X
   [[-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]],

    [[-1, -1, -1],
     [0, 0, 0],
     [1, 1, 1]
    ],

    [[-1, -1, -1],
     [0, 0, 0],
     [1, 1, 1]
    ]
]),

np.array([                       #Prewitt Y
   [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]],

    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]
    ],

    [[-1, -1, -1],
     [-1, -1, -1],
     [-1, -1, -1]
    ]
]),

np.array([                       #Sobel X
   [[-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]],

    [[-2, -2, -2],
     [0, 0, 0],
     [2, 2, 2]
    ],

    [[-1, -1, -1],
     [0, 0, 0],
     [1, 1, 1]
    ]
]),

np.array([                       #Sobel Y
   [[1, 1, 1],
    [2, 2, 2],
    [1, 1, 1]],

    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]
    ],

    [[-1, -1, -1],
     [-2, -2, -2],
     [-1, -1, -1]
    ]
]),

np.array([                       # Robert X
   [[0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]
    ],

    [[-1, -1, -1],
     [0, 0, 0],
     [0, 0, 0]
    ]
]),

np.array([                       # Robert Y
   [[1, 1, 1],
    [0, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
     [-1, -1, -1],
     [0, 0, 0]
    ],

    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]
    ]
]),

np.array([                       # First Derivative Filter X
   [[0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]],

    [[-1, -1, -1],
     [1, 1, 1],
     [0, 0, 0]
    ],

    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]
    ]
]),

np.array([                       # First Derivative Filter Y1
   [[0, 0, 0],
    [-1, -1, -1],
    [0, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]
    ],

    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]
    ]
]),

np.array([                       # First Derivative Filter Y2
   [[0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]
    ],

    [[0, 0, 0],
     [-1, -1, -1],
     [0, 0, 0]
    ]
]),
)
'''print(kernel_cluster[8][:,:,2])
exit()'''
kernel_number= input("Select Kernel 1.Box Filter 2.Prewitt-x 3.Prewitt-y 4.Sobel-x 5.Sobel-y 6.Robert-x 7.Robert-y 8.First Deriv-x 9.First Deriv-y1 10.FirstDeriv-y2")
paddingtype=input("Select padding type 1.Zero 2.Copy 3.Reflect 4.Wraparound")
input_img=input("1.Gray 2.RGB")

output=ImageProcUtils.conv2(input_option[int(input_img)-1], kernel_cluster[int(kernel_number)-1], int(paddingtype)-1)
input_h, input_w =lena_image.shape[:2]
output_h, output_w= output.shape[:2]


cv2.imshow('Filtered Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()  # close the display window on clicking esc