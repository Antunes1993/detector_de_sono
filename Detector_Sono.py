#Processamento de imagem básico
import numpy as np
import cv2
from imutils import resize
import datetime 
import dlib
from numpy.core.einsumfunc import _einsum_dispatcher 
from scipy.spatial import distance as dist

capture = cv2.VideoCapture(0)

#Fontes de texto
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 0, 0)
thickness = 1

#Posicionamento de mensagens 
org = (450, 20) 
org2 = (20, 20)
org3 = (20, 40)

#Classificador
classificador_dlib_68 = "C:\\Users\\feoxp7\\Desktop\\ESTUDO\\detector_de_sono\\classificadores\\shape_predictor_68_face_landmarks.dat"
classificador_dlib = dlib.shape_predictor(classificador_dlib_68)
detector_face = dlib.get_frontal_face_detector()

#Pontos fiduciais
FACE = list(range(17,68))
FACE_COMPLETA = list(range(0,68))
LABIO = list(range(48,61))
SOMBRANCELHA_DIREITA = list(range(17,22))
SOMBRANCELHA_ESQUERDA = list(range(22,27))
OLHO_DIREITO = list(range(36,42))
OLHO_ESQUERDO = list(range(42,48))
NARIZ = list(range(27,35))
MANDIBULA = list(range(0,17))

#Variaveis de controle
modo_apresentacao = False

def main():
    while(True):
        ret, frame = capture.read()
        #Conversão espaço de cores (escala de cinza e espaço de cores hsv)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        
        #Inserção de data
        now = datetime.datetime.now()
        frame = cv2.putText(frame, now.strftime("%d-%m-%Y %H:%M:%S"), org, font, fontScale, color, thickness, cv2.LINE_AA)

        #Detector de Faces
        retangulos = detector_face(frame, 1)

        if (len(retangulos)) > 0:
            frame = detector_faces(frame, retangulos, modo_apresentacao)
        
        #Pontos fiduciais
        if (len(retangulos)) > 0:
            frame, marcos_obtidos = marcos_faciais(frame, retangulos, modo_apresentacao)

        
        #Casca convexa 
        if (len(retangulos)) > 0 and (len(marcos_obtidos)) > 0:
            frame = anotar_marcos_casca_convexa(frame, marcos_obtidos, retangulos, modo_apresentacao)
        
        if (len(marcos_obtidos)) > 0:
            valor_olho_esquerdo = aspecto_razao_olhos(marcos_obtidos[0][OLHO_ESQUERDO])
            valor_olho_direito = aspecto_razao_olhos(marcos_obtidos[0][OLHO_DIREITO])
            valor_labios = aspecto_razao_boca(marcos_obtidos[0][LABIO])

            print("Olho esquerdo: ", valor_olho_esquerdo)
            print("Olho direito: ", valor_olho_direito)
            print("Labios: ", valor_labios)
            #frame = cv2.putText(frame, str(valor_labios), org2, font, fontScale, color, thickness, cv2.LINE_AA)

            if valor_labios > 0.5 or valor_olho_esquerdo < 0.25 or valor_olho_direito < 0.25:
                frame = cv2.putText(frame, "Detectado reducao de atencao causada por cansaco", org3, font, fontScale, color, thickness, cv2.LINE_AA)

        #Exibição do frame      
        cv2.imshow('frame',frame)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()



def detector_faces(frame, retangulos,modo_apresentacao):
    if (modo_apresentacao == True):
        for k, d in enumerate(retangulos):        
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
    return frame

def marcos_faciais(frame,retangulos, modo_apresentacao):
    marcos = []
    if len(retangulos) > 0:
        for ret in retangulos:
            marcos.append(np.matrix([[p.x, p.y] for p in classificador_dlib(frame, ret).parts()]))
    
    if len(marcos) > 0:
        for marco in marcos:
            for idx, ponto in enumerate(marco):
                centro= (ponto[0,0], ponto[0,1])
                if (modo_apresentacao == True):
                    cv2.circle(frame, centro, 2, (45, 255, 255), -1)
                #cv2.putText(frame, str(idx), (centro[0] - 10, centro[1] - 10), font, fontScale, color, thickness, cv2.LINE_AA)

    return frame, marcos

def aspecto_razao_olhos(pontos_olhos):
    a = dist.euclidean(pontos_olhos[1], pontos_olhos[5])
    b = dist.euclidean(pontos_olhos[2], pontos_olhos[4])
    c = dist.euclidean(pontos_olhos[0], pontos_olhos[3])

    aspecto_razao = (a + b) / (2.0 * c)
    return aspecto_razao

def anotar_marcos_casca_convexa(frame, marcos, retangulos, modo_apresentacao):
    if(len(retangulos)) == 0:
        return None 
    
    for idx, ret in enumerate(retangulos): 
        marco = marcos[idx]
        pontos_olho_esquerdo = cv2.convexHull(marco[OLHO_ESQUERDO])
        
        pontos_olho_direito = cv2.convexHull(marco[OLHO_DIREITO])

        pontos_labio = cv2.convexHull(marco[LABIO])

        if(modo_apresentacao == True):
            cv2.drawContours(frame, [pontos_olho_esquerdo], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [pontos_olho_direito], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [pontos_labio], -1, (0, 255, 0), 2)

    return frame

def aspecto_razao_boca(pontos_boca):
    a = dist.euclidean(pontos_boca[3], pontos_boca[9])
    b = dist.euclidean(pontos_boca[2], pontos_boca[10])
    c = dist.euclidean(pontos_boca[4], pontos_boca[8])
    d = dist.euclidean(pontos_boca[0], pontos_boca[6])

    aspecto_razao = (a + b + c) / (3.0 * d)

    return aspecto_razao


main()