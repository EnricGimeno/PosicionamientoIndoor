#LIBRERIAS CONEXION USB
import usb
import gudev
import glib
import pynotify
import sys

##LIBERIAS TKINTER Y PROGRAMA
import time
import threading
import numpy as np
import numpy
import Tkinter
import Image, ImageTk
import tkMessageBox
import math

## Lectura multiples ficheros
import glob
import errno

import cv2

'''
            ROUT TRACKING.

This script determine 

USAGE:

    - First step: 


'''


def start_cam1():
    t1 = threading.Thread(target=show_cam1)
    t1.start()

def show_cam1():
    global n
    global cam1
    global C1_L1,C1_L2,C1_L3,C1_L4,C1_L5,C1_L6,C1_L7,C1_L8,C1_L9,C1_L10,C1_L11
    global point1
    global point2
    global coord_3D_file
    point2 = None

    ## CREACION DEL FICHERO DE COORDENADAS
    coord_3D_file = open('Coordenadas_3D.csv', 'wt')
    coord_3D_file.write('U_img1, V_img1, U_img2, V_img2, X_Terreno, Y_Terreno, Z_Terreno \n')
    

    # LECTURA DEL FICHERO DE CALIBRACION
    calib_file_1 = open (camera1_calibration, 'rt')
    calib_data_1 = calib_file_1.readlines()
    calib_file_1.close()
    print camera1_calibration

    for calib1 in calib_data_1:
        C1_L1,C1_L2,C1_L3,C1_L4,C1_L5,C1_L6,C1_L7,C1_L8,C1_L9,C1_L10,C1_L11 = calib1.strip().split(',')
        
    

    cam1 = cv2.VideoCapture(1)
    # Modify dimensions picture
    ret1 = cam1.set(3,1280.0)
    ret1 = cam1.set(4,720.0)
    #ret1 = cam1.set(cv2.CAP_PROP_FPS,1)

    
    quit_bttn.config(state='disabled')
    start1_bttn.config(state='disabled')
    stop1_bttn.config(state='normal')
    
    n += 1
    
    print 'Wait 5 seconds while the camera is stabilized...'
    end_time = time.time() + 0.001388889 * 3600
    while(time.time() < end_time ):
    # Tomamos la imagen de fondo y Aplicamos un filtro de Gauss para eliminar
    #los ruidos de una imagen.
        ret, fondo = cam1.read()
        fondo = cv2.GaussianBlur(fondo.copy(),(3,3),0)
        
    print ' Picture of the Background camera 1 taken...'
    # Capture frames...
    while (cam1.isOpened()):
        ret, frame = cam1.read()
        image = cv2.GaussianBlur(frame,(3,3),0)
        # Calculo de la diferencia absoluta entre dos arrays o array y escalar
        # esta restando la imagen (pasada por el filtro) menos el fondo
        mascara=image.copy()
        cv2.absdiff(image,fondo,mascara)
        #Convierte una imagen de un espacio de color a otro. En este caso a una
        #escala de GRIS
        gray = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
        #Se emplea un Threshold Binary
        #Para diferenciar los píxeles que nos interesa del resto + Info: Programa: Nuestro_background
        ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)

        ## Obtenemos los parametros de la barra
        tdilated = barraDilated.get()
        terode = barraErode.get()
        tareamin = barraMinArea.get()   

        ## aumenta la imagen. 
        #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        dilated = cv2.dilate(thresh1,kernel,iterations = tdilated)
        # Erode es lo contrario a dilated
        # con esto podemos obtener otro contorno en la imagen porque es como hacer una resta
        erode = cv2.erode(dilated,kernel,iterations = terode)

        ## Encotramos el contorno del los objectos
        _,contorno,heir=cv2.findContours(erode,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        for c in contorno:
            area = cv2.contourArea(c)
            if area > tareamin and area < 900000:
                ## Dibujamos los contornos
                cv2.drawContours(image, c, -1, (0,255,0),3)
                # x,y es las coordenadas de la parte superior izquierda
                x,y,w,h = cv2.boundingRect(c)
                rect = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
                ## Punto superior del contorno
                topmost = tuple(c[c[:,:,1].argmin()][0])
                pnt = cv2.circle(image,(topmost[0],topmost[1]),5,(0,0,255),thickness=-1)
                point1 = topmost[0],topmost[1]
                if draw == True:
                    point_3d(point1, point2)

        ##CV2
        #cv2.imshow("CV2", image)
        ##RESIZED IMAGE
        r = 460.0 / image.shape[1]
        dim = (460, int(image.shape[0] * r))
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        ## CARGA DE LA IMAGEN
        b,g,r = cv2.split(resized)
        rgb_img = cv2.merge([r,g,b]) 
        img = Image.fromarray(rgb_img)  
        photo = ImageTk.PhotoImage(image=img)
        #cv2.imshow("CAMERA 1", resized)
        #cv2.waitKey(10)
        cnvs1.create_image(0, 0, anchor='nw', image=photo)
        #root.update()
        cnvs1.update()

    print 'Camera #1 released'
    

def stop_cam1():
    global n

    n -= 1
    
    cam1.release()
    
    root.update_idletasks()
    start1_bttn.config(state='normal')
    stop1_bttn.config(state='disabled')
    
    
    if n == 0:
        quit_bttn.config(state='normal')

    coord_3D_file.close()


def start_cam2():
    t2 = threading.Thread(target=show_cam2)
    t2.start()

def show_cam2():
    global n
    global cam2
    global C2_L1,C2_L2,C2_L3,C2_L4,C2_L5,C2_L6,C2_L7,C2_L8,C2_L9,C2_L10,C2_L11
    global point1
    global point2
    point1 = None

    # LECTURA DEL FICHERO DE CALIBRACION 
    calib_file_2 = open (camera2_calibration, 'rt')
    calib_data_2 = calib_file_2.readlines()
    calib_file_2.close()
    print camera2_calibration

    for calib2 in calib_data_2:
        C2_L1,C2_L2,C2_L3,C2_L4,C2_L5,C2_L6,C2_L7,C2_L8,C2_L9,C2_L10,C2_L11 = calib2.strip().split(',')
    

    
    cam2 = cv2.VideoCapture(2)
    # Modify dimensions picture
    ret2 = cam2.set(3,1280.0)
    ret2 = cam2.set(4,720.0)
    #ret2 = cam2.set(5,1)

    quit_bttn.config(state='disabled')
    start2_bttn.config(state='disabled')
    stop2_bttn.config(state='normal')

    n += 1
    
    print 'Wait 5 seconds while the camera is stabilized...'
    end_time = time.time() + 0.001388889 * 3600

    while(time.time() < end_time ):
    # Tomamos la imagen de fondo y Aplicamos un filtro de Gauss para eliminar
    #los ruidos de una imagen.
        ret, fondo = cam2.read()
        fondo = cv2.GaussianBlur(fondo.copy(),(3,3),0)

    print ' Picture of the Background camera 2 taken...'
    # Capture frames...
    while (cam2.isOpened()):
        ret, frame = cam2.read()
        image = cv2.GaussianBlur(frame,(3,3),0)
        # Calculo de la diferencia absoluta entre dos arrays o array y escalar
        # esta restando la imagen (pasada por el filtro) menos el fondo
        mascara=image.copy()
        cv2.absdiff(image,fondo,mascara)
        #Convierte una imagen de un espacio de color a otro. En este caso a una
        #escala de GRIS
        gray = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
        #Se emplea un Threshold Binary
        #Para diferenciar los píxeles que nos interesa del resto + Info: Programa: Nuestro_background
        ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)

        ## Obtenemos los parametros de la barra
        tdilated = barraDilated.get()
        terode = barraErode.get()
        tareamin = barraMinArea.get()

        ## aumenta la imagen. 
        #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        dilated = cv2.dilate(thresh1,kernel,iterations = tdilated)
        # Erode es lo contrario a dilated
        # con esto podemos obtener otro contorno en la imagen porque es como hacer una resta
        erode = cv2.erode(dilated,kernel,iterations = terode)

        ## Encotramos el contorno del los objectos
        _,contorno,heir=cv2.findContours(erode,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        for c in contorno:
            area = cv2.contourArea(c)
            if area > tareamin and area < 900000:
                ## Dibujamos los contornos
                cv2.drawContours(image, c, -1, (0,255,0),3)
                # x,y es las coordenadas de la parte superior izquierda
                x,y,w,h = cv2.boundingRect(c)
                rect = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
                ## Punto superior del contorno
                topmost = tuple(c[c[:,:,1].argmin()][0])
                pnt = cv2.circle(image,(topmost[0],topmost[1]),5,(0,0,255),thickness=-1)
                point2 = topmost[0],topmost[1]
                if draw == True:
                    point_3d(point1, point2)
                

        ##RESIZED IMAGE
        r = 460.0 / image.shape[1]
        dim = (460, int(image.shape[0] * r))
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        ## CARGA DE LA IMAGEN
        b,g,r = cv2.split(resized)
        rgb_img = cv2.merge([r,g,b]) 
        img = Image.fromarray(rgb_img)  
        photo = ImageTk.PhotoImage(image=img)
        #cv2.imshow("CAMERA 2", resized)
        cnvs2.create_image(0, 0, anchor='nw', image=photo)
        #root.update()
        cnvs2.update()


    print 'Camera #2 released'




def stop_cam2():
    global n

    n -= 1
    
    cam2.release()
    
    root.update_idletasks()
    start2_bttn.config(state='normal')
    stop2_bttn.config(state='disabled')
    
    if n == 0:
        quit_bttn.config(state='normal')


def point_3d(point1, point2):
    global vector_puntos
    global U_img1,U_img2,V_img1,V_img2
    
    
    if (point1 == None or point2 == None):
        pass
    
    else:
        ## COORDENADAS IMAGEN
        U_img1 = float(point1[0])
        V_img1 = float(point1[1])
        U_img2 = float(point2[0])
        V_img2 = float(point2[1])
        
        ## CAMERA 1 
        a1 = float(C1_L1) - U_img1*float(C1_L9)
        b1 = float(C1_L2) - U_img1*float(C1_L10)
        c1 = float(C1_L3) - U_img1*float(C1_L11)
        d1 = U_img1*1.0 - float(C1_L4) 
        a2 = float(C1_L5) - V_img1*float(C1_L9)
        b2 = float(C1_L6) - V_img1*float(C1_L10)
        c2 = float(C1_L7) - V_img1*float(C1_L11)
        d2 = V_img1*1.0 - float(C1_L8) 

        ## CAMERA 2 
        p1 = float(C2_L1) - U_img2*float(C2_L9)
        q1 = float(C2_L2) - U_img2*float(C2_L10)
        r1 = float(C2_L3) - U_img2*float(C2_L11)
        s1 = U_img2*1.0 - float(C2_L4) 
        p2 = float(C2_L5) - V_img2*float(C2_L9)
        q2 = float(C2_L6) - V_img2*float(C2_L10)
        r2 = float(C2_L7) - V_img2*float(C2_L11)
        s2 = V_img2*1.0 - float(C2_L8)  

        ## MATRIZ A
        a = []
        a.append([a1,b1,c1])
        a.append([a2,b2,c2])
        a.append([p1,q1,r1])
        a.append([p2,q2,r2])

        ## MATRIZ B
        b = []
        b.append([d1])
        b.append([d2])
        b.append([s1])
        b.append([s2])

        x = numpy.linalg.lstsq(a, b)[0].ravel().tolist()
        #print x

        vector_puntos.append(x)
        if len(vector_puntos) == 20:
            coordX = 0
            coordY = 0
            for i in vector_puntos:
                coordX = coordX + i[0]
                coordY = coordY + i[1]
                coordZ = coordY + i[2]

            coordX = coordX/len(vector_puntos)
            coordY = coordY/len(vector_puntos)
            coordZ = coordZ/len(vector_puntos)
            puntoPintar = [coordX,coordY,coordZ]
            vector_puntos =[]
            pintar_punto(puntoPintar)
            
            

def pintar_punto(x):
    
    print 'Punto Pintado'
    xtrans, ytrans = transformacion_punto(x)
    ytrans = height_map - ytrans
    cnvs3.create_rectangle(xtrans-2,ytrans-2, xtrans+2,ytrans+2,fill = "yellow", outline = "yellow")
    coord_3D_file.write(str(U_img1) + ',' +  str(V_img1)+ ',' +str(U_img2) + ',' +  str(V_img2) + ',' + 
                            str(x[0]) + ',' + str(x[1]) + ',' + str(x[2]) +'\n')
    
    





def quitw(e):

##    root.update()    
    root.destroy()

def USB_camera1():
    global loop
    global camera1
    global camera2
    camera1 = True
    camera2 = False
    client = gudev.Client(["usb/usb_device"])
    client.connect("uevent", callback, None)

    loop = glib.MainLoop()
    loop.run()

def USB_camera2():
    global loop
    global camera1
    global camera2
    camera1 = False
    camera2 = True
    client = gudev.Client(["usb/usb_device"])
    client.connect("uevent", callback, None)

    loop = glib.MainLoop()
    loop.run()


def callback(client, action, device, user_data):
    global camera1_calibration
    global camera2_calibration
    global camera_utilizada
    
    busses = usb.busses()
    for bus in busses:
        devices = bus.devices
        for dev in devices:
            handle = dev.open()
            for i in range(0,len(cameras)):
                idVendedor = cameras[i][0]
                idProducto = cameras[i][1]
                Nombre = cameras[i][2]
                nombre_fichero_calibracion = cameras[i][3]

                idVendedorUSB = ("{:04x}".format(dev.idVendor))
                idProductoUSB = ("{:04x}".format(dev.idProduct))

                if (camera1 == True)and(idVendedor==idVendedorUSB) and (idProducto==idProductoUSB):
                    print Nombre
                    if action == "add":
                        n = pynotify.Notification("USB Device Added", "%s is now connected "
                                  "to your system and calibration file is associated "
                                  "to the camera."% (Nombre))
                        n.show()
                    camera1_calibration = nombre_fichero_calibracion + '.csv'
                    camera_utilizada = Nombre
                    
                if (camera2 == True)and(idVendedor==idVendedorUSB) and (idProducto==idProductoUSB)and(camera_utilizada!=Nombre):
                    print Nombre
                    if action == "add":
                        n = pynotify.Notification("USB Device Added", "%s is now connected "
                                  "to your system and calibration file is associated "
                                  "to the camera."% (Nombre))
                        n.show()
                    camera2_calibration = nombre_fichero_calibracion + '.csv'
                    start1_bttn.config(state='normal')
                    start2_bttn.config(state='normal')
                    
    
        
        loop.quit()
    

### ROOM FILE AND PLOT CANVAS FUNCTIONS
## LECTURA DEL FICHERO DE COORDENADAS DE LA HABITACION
def read_file_room(e):
  global Coor_hab
  
  # LECTURA DEL FICHERO DE COORDENADAS DE LA HABITACION
  room_file = open ('Room_Real.csv', 'rt')
  room_data = room_file.readlines()
  room_file.close()

  Coor_hab = []
  cont = 1
  for room in room_data:
    punto = room.strip().split(',')
    x = punto[0]
    y = punto[1]
    Coor_hab.append(punto)
    ##cnvs3.create_oval(float(x),float(y), float(x)+5,float(y)+5,fill = "orange", outline = "orange")
    ##cnvs3.create_text(float(x),float(y), text=str(cont), fill='orange')
    cont = cont + 1
  ##Añadimos los dos primeros puntos al final
  for p in range(0,2):
    Coor_hab.append(Coor_hab[p])  
  print Coor_hab  
  corners_room()

def bbox (pol):
    xmin = pol[0][0]
    xmax = pol[0][0]
    ymin = pol[0][1]
    ymax = pol[0][1]
    for pnt in pol:
        if pnt[0] < xmin:
            xmin = pnt[0]
        elif pnt[0] > xmax:
            xmax = pnt[0]
        if pnt[1] < ymin:
            ymin = pnt[1]
        elif pnt[1] > ymax:
            ymax = pnt[1]
    return [xmin,ymin,xmax,ymax]    
      
def intersect(p1,p2,p3,p4):

    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])
    x3 = float(p3[0])
    y3 = float(p3[1])
    x4 = float(p4[0])
    y4 = float(p4[1])

    #DENOMINADOR
    d = (y2 - y1)*(x4 -x3) - (x2-x1) * (y4 - y3)

    if d == 0:
         return None

    na = (x1 - x3)*(y4 - y3) - (y1 - y3)*(x4 - x3)
    nb = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)

    ua = na/d
    ub = nb/d

    x_int = x1 + ua*(x2 - x1)
    y_int = y1 + ua*(y2 - y1)

    if ua <= 1.0 and ua >= 0.0 and ub <= 1.0 and ub >= 0.0:
        return [x_int, y_int, True]

    else:
        return [x_int, y_int, False]

## Elementos de la habitacion

def elements_room():
    path = 'Elementos*.csv'   
    files = glob.glob(path)
    for name in files: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        vector_elementos = []
        try:
            with open(name) as f: 
                elementos_data = f.readlines()
                for elemento in elementos_data:
                    puntos = elemento.strip().split(',')
                    xtrans, ytrans = transformacion_punto(puntos)
                    ytrans = height_map - ytrans
                    cnvs3.create_rectangle(xtrans-2,ytrans-2, xtrans+2,ytrans+2,fill = "white")
                    vector_elementos.append([xtrans,ytrans])

                vector_elementos.append([vector_elementos[0][0],vector_elementos[0][1]])
                print 'VECTOR'
                print vector_elementos

                ## Ploteado de las lineas
                for l in range (0,len(vector_elementos)-1):
                    cnvs3.create_line(vector_elementos[l][0],vector_elementos[l][1],
                      vector_elementos[l+1][0],vector_elementos[l+1][1],fill='white')
                
                    
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.
   
    

## Calculo de las esquinas de la habitacion
def corners_room():
    global corners
    corners =[]
    cont = 0
    num_puntos = (len(Coor_hab)-2)/2
    for i in range(0,(num_puntos)):
        corner = intersect(Coor_hab[cont],Coor_hab[cont+1],Coor_hab[cont+2],Coor_hab[cont+3])  
        corners.append([corner[0],corner[1]])
        cont += 2
    print corners
    room_transformation()
    

def room_transformation():
  global corners_rotated
  global alfa
  global sf
  global tx, ty
  global cx,cy
  corners_rotated = []  

  ## Globals
  width_map = 460
  height_map = 258
  
  ## Dimensiones maximas
  dxmax = corners[1][0] - corners[3][0]
  dymax = corners[0][1] - corners[2][1]
  
  ##Distancia entre los dos primeros puntos
  dist = math.sqrt((corners[1][0]-corners[2][0])*(corners[1][0]-corners[2][0]) +
                   (corners[1][1]-corners[2][1])*(corners[1][1]-corners[2][1]))
  print dist

  ## Calculo del angulo
  dx= corners[1][0] - corners[2][0]
  dy = corners[1][1] - corners[2][1]
  
  if corners[1][1] > corners[2][1]:
    alfa = (math.atan(dy/dx))*-1
  if corners[1][1] < corners[2][1]:
    alfa = (math.atan(dy/dx))
  if corners[1][1] == corners[0][1]:
    alfa = 0
    
  print alfa  
    
  for i in range(0, len(corners)):
    #cnvs3.create_oval(corners[i][0],corners[i][1], corners[i][0]+5,corners[i][1]+5,fill = "yellow", outline = "yellow")
    #cnvs3.create_text(corners[i][0],corners[i][1], text=str(i), fill='yellow')

    ## Punto rotacion
    x0 = corners[2][0]
    y0 = corners[2][1]
    ## Punto a rotar
    x = corners[i][0]
    y = corners[i][1]

    ##Coordenadas rotadas
    xr = x0 + (x - x0)*math.cos (alfa) - (y - y0)*math.sin (alfa)
    yr = y0 + (x - x0)*math.sin (alfa) + (y - y0)*math.cos (alfa)
    #cnvs3.create_oval(xr,yr, xr+5,yr+5,fill = "blue", outline = "blue")
    #cnvs3.create_text(xr, yr, text=str(i), fill='blue')
    corners_rotated.append([xr,yr])

  ## Comprobacion dimensiones (deben ser iguales). Dos primeros puntos
  dist = math.sqrt((corners_rotated[1][0]-corners_rotated[2][0])*(corners_rotated[1][0]-corners_rotated[2][0]) +
                   (corners_rotated[1][1]-corners_rotated[2][1])*(corners_rotated[1][1]-corners_rotated[2][1]))
  print dist
 

  ## Calculo del centro aproximado mediante una bounding box
  xmin,ymin,xmax,ymax = bbox(corners_rotated)
  cx = 0.5*(xmin + xmax)
  cy = 0.5*(ymin + ymax)
  #cnvs3.create_oval(cx, cy,cx+5,cy+5,fill = "green", outline = "green")

  ## Coordenadas del centro del canvas
  cx_cnvs = width_map/2
  cy_cnvs = height_map/2

  ## Calculo de la traslacion
  tx = cx - cx_cnvs
  ty = cy - cy_cnvs

  coor_trans = []
  for j in range (0, len(corners_rotated)):
    xr = corners_rotated[j][0]
    yr = corners_rotated[j][1]
    ##Coordenadas trasladadas
    xt = xr - tx
    yt = yr - ty
    coor_trans.append([xt,yt])
    #cnvs3.create_oval(xt,yt, xt+5,yt+5,fill = "red", outline = "red")
    #cnvs3.create_text(xt, yt, text=str(j), fill='red')

  ## Calculo de los factores de escala
  dxs = xmax - xmin
  dys = ymax - ymin
  sx = 0.9*width_map/dxs
  sy = 0.9*height_map/dys

  if sx < sy:
    sf = sx
  else:
    sf = sy
  print sf

  ##Centro trasladado para aplicar el escalado
  cx = cx - tx
  cy = cy - ty

  coor_sca = []
  ##Coordenadas scaladas
  for c in range (0,len(coor_trans)):
    xt = coor_trans[c][0]
    yt = coor_trans[c][1]
    
    xs = cx + sf*(xt-cx)
    ys = cy + sf*(yt-cy)
    ys = height_map -ys
    cnvs3.create_rectangle(xs-2,ys-2, xs+2,ys+2,fill = "white")
    coor_sca.append([xs,ys])
    

  ## Ploteado
  for l in range (0,len(coor_sca)-1):
    
    cnvs3.create_line(coor_sca[l][0],coor_sca[l][1],
                      coor_sca[l+1][0],coor_sca[l+1][1],fill='white')

  cnvs3.create_line(coor_sca[0][0],coor_sca[0][1],
                      coor_sca[-1][0],coor_sca[-1][1],fill='white')

  ## UNA VEZ CARGADA LA HABITACION
  ## CARGAMOS SUS ELEMENTOS
  elements_room()
  
    

def transformacion_punto(punto):
  ## Punto rotacion
  x0 = corners[2][0]
  y0 = corners[2][1]
  x = float(punto[0])
  y = float(punto[1])

  ## COORDENADAS ROTADAS
  xr = x0 + (x - x0)*math.cos (alfa) - (y - y0)*math.sin (alfa)
  yr = y0 + (x - x0)*math.sin (alfa) + (y - y0)*math.cos (alfa)

  ## COORDENADAS TRASLADADAS
  xt = xr - tx
  yt = yr - ty
  
  ## COORDENADAS SCALADAS
  xs = cx + sf*(xt-cx)
  ys = cy + sf*(yt-cy)
  

  return xs, ys

    
def delete_cnvs(e):
    cnvs3.delete('all')

def start_draw(e):
    global draw
    draw = True
    print ' Drawing'

def stop_draw(e):
    global draw
    draw = False
    print ' Stop Drawing'
    

if __name__ == '__main__':
    print __doc__
    global cameras
    global draw
    global vector_puntos 
    draw = False
    vector_puntos =[]

    ## Para que muestre el mensaje de conexion USB
    if not pynotify.init("USB Device Notifier"):
        sys.exit("Couldn't connect to the notification daemon!")

    #LECTURA DEL FICHERO DE CAMARAS
    input_file = open ('Cameras.csv', 'rt')
    input_data = input_file.readlines()
    input_file.close()
    cameras = []
    for cam in input_data:
        idVendedor, idProduct, Nombre, Nom_fich_cal = cam.strip().split(',')
        cameras.append([idVendedor, idProduct, Nombre, Nom_fich_cal])
    #print cameras    
    
    # GLOBALS
    n = 0
    width_cam01 = 460
    height_cam01 = 258
    width_cam02 = 460
    height_cam02 = 258
    width_map = 460
    height_map = 258
    
    

    root = Tkinter.Tk()
    root.title('ROUT TRACKING ')
    
    # BOORDE NORMAL Tkinter.GROOVE
    
    frm1 = Tkinter.LabelFrame(root, text=' Camera #1  ', padx=10, pady=10, relief= Tkinter.GROOVE)
    frm2 = Tkinter.LabelFrame(root, text=' Camera #2  ', padx=10, pady=10, relief= Tkinter.GROOVE)
    frm3 = Tkinter.LabelFrame(root, text=' Route tracking #3  ', padx=10, pady=10, relief= Tkinter.GROOVE)
    frm4 = Tkinter.LabelFrame(root, text=' Configuration #4  ', padx=10, pady=10, relief= Tkinter.GROOVE)
    frm5 = Tkinter.LabelFrame(frm4, text=' Camera 1:  ', padx=5, pady=5, relief= Tkinter.FLAT)
    frm6 = Tkinter.LabelFrame(frm4, text=' Camera 2:  ', padx=5, pady=5, relief= Tkinter.FLAT)
    frm7 = Tkinter.LabelFrame(frm4, text='', padx=5, pady=5, relief= Tkinter.FLAT, borderwidth= 0)
    frm8 = Tkinter.LabelFrame(frm4, text='', padx=5, pady=5, relief= Tkinter.FLAT, borderwidth= 0)
    
                             
    

    cnvs1 = Tkinter.Canvas(frm1, width=width_cam01, height=height_cam01, bg='black')
    cnvs2 = Tkinter.Canvas(frm2, width=width_cam02, height=height_cam02, bg='black')
    cnvs3 = Tkinter.Canvas(frm3, width=width_map, height=height_map, bg='black')
    
    ## ELEMENTOS DEL FRAME CONFIGURACION
    start1_bttn = Tkinter.Button(frm5, text='Start', width=6, command=start_cam1)
    stop1_bttn  = Tkinter.Button(frm5, text='Stop', width=6, command=stop_cam1, state='disabled')
    start2_bttn = Tkinter.Button(frm6, text='Start', width=6, command=show_cam2)
    stop2_bttn  = Tkinter.Button(frm6, text='Stop', width=6, command=stop_cam2, state='disabled')
    quit_bttn = Tkinter.Button(frm8, text='Quit', width=10)
    start_draw_bttn = Tkinter.Button(frm8, text='Start draw', width=10)
    stop_draw_bttn = Tkinter.Button(frm8, text='Stop draw', width=10)
    clear_bttn = Tkinter.Button(frm8, text='Clear', width=10)
    room_bttn = Tkinter.Button(frm8, text='Room', width=10)
    start1_bttn.grid(row=0, column=0, padx=20)
    stop1_bttn.grid(row=0, column=1, padx=20)
    start2_bttn.grid(row=0, column=0, padx=20)
    stop2_bttn.grid(row=0, column=1, padx=20)
    quit_bttn.grid(row=1, column=2, pady= 5 )
    start_draw_bttn.grid(row=0, column=0, pady=5 )
    stop_draw_bttn.grid(row=0, column=1, pady=5 )
    clear_bttn.grid(row=0, column=2, pady=5 )
    room_bttn.grid(row=1, column=0, pady=5 )
    start_draw_bttn.config(state='normal')
    stop_draw_bttn.config(state='normal')

    Dilated = Tkinter.StringVar()
    Dilated.set('Dilated ')

    lbl_Dilated = Tkinter.Label(frm7, textvariable = Dilated)
    lbl_Dilated.grid(row=0, column=0)

    Erode = Tkinter.StringVar()
    Erode.set('Erode ')

    lbl_Erode = Tkinter.Label(frm7, textvariable = Erode)
    lbl_Erode.grid(row=2, column=0)

    Area = Tkinter.StringVar()
    Area.set('Area ')

    lbl_Area = Tkinter.Label(frm7, textvariable = Area)
    lbl_Area.grid(row=4, column=0)


    barraDilated = Tkinter.Scale(frm7, from_=0, to=50,orient=Tkinter.HORIZONTAL, length=300)
    barraDilated.grid(row=0, column=1)
    barraErode = Tkinter.Scale(frm7, from_=0, to=50,orient=Tkinter.HORIZONTAL, length=300)
    barraErode.grid(row=2, column=1)
    barraMinArea = Tkinter.Scale(frm7, from_=0, to=100000,orient=Tkinter.HORIZONTAL, length=300)
    barraMinArea.grid(row=4, column=1)

    cnvs1.grid(row=0, column=0, rowspan=4)
    cnvs2.grid(row=0, column=0, rowspan=4)  
    cnvs3.grid(row=0, column=0, rowspan=4)  
   
    frm1.grid(row=0, column=0, padx=10, pady=10)
    frm2.grid(row=1, column=0, padx=10, pady=10)    
    frm3.grid(row=0, column=1, padx=10, pady=10)
    frm4.grid(row=1, column=1, padx=10, pady=10)
    frm5.grid(row=0, column=0, padx=5, pady=5)
    frm6.grid(row=1, column=0, padx=5, pady=5)
    frm7.grid(row=2, column=0, padx=5, pady=5)
    frm8.grid(row=3, column=0, padx=5, pady=5)

    start1_bttn.config(state='disabled')
    start2_bttn.config(state='disabled')

    quit_bttn.bind('<Button-1>',quitw)
    clear_bttn.bind('<Button-1>',delete_cnvs)
    room_bttn.bind('<Button-1>',read_file_room)
    start_draw_bttn.bind('<Button-1>',start_draw)
    stop_draw_bttn.bind('<Button-1>',stop_draw)

    tkMessageBox.showinfo("CONNECTION", "Disconnect all the cameras")
    tkMessageBox.showinfo("CAMERA", "Connect the first camera")
    USB_camera1()
    tkMessageBox.showinfo("CAMERA", "Connect the second camera")
    USB_camera2()
    

    root.mainloop()



