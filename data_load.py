# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:37:26 2018

@author: Lina
"""



import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import html

data_dim = 2
timesteps = 2000

#tree = ET.parse("xml_trial2.txt")
#root = tree.getroot()

X=[] 
Y=[]

def num_to_vector(r):
    l = [0 for _ in range(81)]
    l[r] = 1
    return l
    

line_count=0  
n=0  
d=0  
#for root, directories, filenames in os.walk('C:\\Users\\Abeer Eisa\\Desktop\\EEE\\5\\Graduation Project\\IAM\\original'):
for root, directories, filenames in os.walk('ORIGINAL/'):
    for filename in filenames:
        d=d+1
        
#        print(os.path.join(root,filename))
        a=os.path.join(root,filename)
        tree=ET.parse(a)
        r = tree.getroot()
        
        
        nums=[]
        k=0
        o=0
        
        alpha='abcdefghijklmnopqrstuvwxyz'
        numbers='0123456789'
        symbols = {':': 63, '"': 64, '%': 65, '/': 66, '.': 67, '!': 68,
                   '?': 69, '(': 70, ')': 71, '#': 72, ';': 73, ',': 74, "'": 75, '&': 76, '+': 77, '-': 78, ' ':79}
        symbols_inverse = {63:':', 64: '"',65 : '%', 66: '/',  67:'.', 68:'!',
                   69:'?',  70:'(', 71:')', 72:'#', 73:';', 74:',', 75:"'", 76:'&',  77:'+', 78:'-', 79:' '}
        SOS_token = 0
        EOS_token = 80
        j=0
        trans = r.find("Transcription")
        for i, textline in enumerate(r.iter('TextLine')):
            o=o+1
            
        words=np.zeros((o,70))
        
        
        for i, textline in enumerate(r.iter('TextLine')):
            text = textline.get('text')
            f=html.unescape(text) #converts the &amp;quote to "
            
#            x= f.split()
            k=k+1
            l=0
            for word in f:
               for z in word:
#                    _+=1
                    if z in alpha.lower(): #lower letters 27 to 52
                        let=(abs(ord(z)-70))
                       
                    elif z in alpha.upper(): #upper letters 1 to 26
                        let=(abs(ord(z)-64))
                        
                    elif z in numbers: #numbers 53 to 62
                        let = ord(z)+5
                        
                    elif z in symbols.keys():
                        let = symbols[z]
                    else:
                        raise Exception('Unexpected char while parsing dataset: {}'.format(z))
#                    for l, c in enumerate(num_to_vector(let)):
                    words[j][l] = let
                    l+=1
            j+=1
#            print(words)
#        print(k)

#        print(k)
#        
#        words=np.asarray(words)
#        print(words.shape)
#        for i in words:
#            print(i)
#            print('\n')
#            print('\n')
#            print('\n')
        
            
            
            
            
            
            
            
        
        stroke_set = r.find("StrokeSet")
        
        strokes = []
        time_stamp = []
        first_time = 0.0
        maxmin=[]
#        maxmin2=[]
#        maxmin3=[]
        
        maxminy=[]
        
        
        
        for stroke_node in stroke_set:
            maxmin.append([])
            maxminy.append([])
            for point in stroke_node:
                
                x = int(point.attrib['x']) 
                y = int(point.attrib['y'])
                timee = float(point.attrib['time'])
                #strokes[-1].append((x,y))
                maxmin[-1].append(x)
                maxminy[-1].append(y)
                strokes.append((x,y))
                if len(time_stamp) == 0:
                        first_time = float(timee)
                        time_stamp.append(0.0)
                else:
                        time_stamp.append( float(timee) - first_time)
        time_listt = []           
        for stroke in r.findall('StrokeSet/Stroke'):
             dumm=[]
            
             for point in stroke.findall('Point'):
                    if len(dumm) == 0:
                        first_time = float(point.get('time'))
                        time_listt.append(0.0)
                        dumm.append(0.0)
                    else:
                        time_listt.append(
                                float(point.get('time')) - first_time)
                        dumm.append(0.0)
        
        
#        maxmin=np.asarray(maxmin)
        
#        print(maxmin3)
        
#        maxmin2=np.delete(maxmin,0,0)
#        maxmin2=np.append(maxmin2,0)
       
#        print(maxmin2)
        
                
        
    
        
            
        
 #normalize:       
        x_elts = [x[0] for x in strokes]
        y_elts = [x[1] for x in strokes]
        #print(x_elts)
        current_Xmax = max(x_elts)
        current_Xmin = min(x_elts)
        current_Ymax = max(y_elts)
        current_Ymin = min(y_elts)
        scale = 1.0 / (current_Ymax - current_Ymin)  #x=6912 y=8798 WhiteboardDescription/DiagonallyOppositeCoords
        
        #print(current_Xmax)
        #print(current_Ymax)
        x_list=[]
        y_list=[]
        for xx_point in x_elts:
            xx_point=((xx_point - current_Xmin)/(current_Xmax-current_Xmin))
            x_list.append(xx_point)
        #print(x_list)
        for yy_point in y_elts:
            yy_point=((yy_point - current_Ymin)/(current_Ymax-current_Ymin))
            y_list.append(yy_point)
            
        #print(y_list)
        combined=[]
        combined= list(zip(x_list , y_list))
 #end normalized       
        #print(combined)
        maxmin2=[]
        maxmin2=maxmin
        maxmin3 = maxmin[1:]
#        print(maxmin3)
        maxminy2=[]
        maxminy2=maxminy
        maxminy3 = maxminy[1:]
        
            
            
     #print(strokes)
#        for stroke in combined:
#            plt.plot(*zip(*combined))
#        plt.gca().invert_yaxis() # gca= Get Current Axis
#        plt.show()    
            
     
#    maxx=[]
#    minn=[]
#    for k in maxmin:
#        maxx.append(max(k))
#        minn.append(min(k))
#    maxx=np.asarray(maxx)
#    minn=np.asarray(minn)
#    minn=np.delete(minn,0,0)
#    minn=np.append(minn,5448)
#    #print(maxx)
#        


        g=1
        c=np.asarray(combined)
        c=np.insert(c, 2, 0, axis=1) #adds third('2') column('axis=1') of zeros
        c=np.insert(c, 3, 0, axis=1)
        for i,val in enumerate(c):
           c[i,3]=time_listt[i]
        
        
        #print(c)
        count=0
        leni=0
    
    
   
                    
    
               
        ind1=0
        ind2=0   
              
#        for x in maxmin3:
#            for u in x:
##                maxmin3[ind1][ind2]=((u - current_Xmin)/(current_Xmax-current_Xmin))
#                ind2+=1
#            ind2=0
#            ind1+=1
    
    
        indy3=0
        indy4=0 
        
                 
        for y in maxminy2:
            for u in y:
                maxminy2[indy3][indy4]=((u - current_Ymin)/(current_Ymax-current_Ymin))
                indy4+=1
                
            indy4=0
            indy3+=1
            
            
       
        
        ind3=0
        ind4=0 
        
                 
        for x in maxmin2:
            for u in x:
                maxmin2[ind3][ind4]=((u - current_Xmin)/(current_Xmax-current_Xmin))
                ind4+=1
                
            ind4=0
            ind3+=1
            
#        
        
#        print(maxmin3)
    
#        for i, j, k, l in zip(maxmin2, maxmin3, maxminy2, maxminy3):
            
        for i, j in zip(maxmin2, maxmin3):
#            print(i[0]-i[-1])
            count+=1
            leni+=len(i)
#            print(max(i), min(j))
#            print(abs(max(i)-min(j)))
#            if abs(max(i)-min(j))>0.023 : #and abs(max(i)-min(j))<0.5:   0.07868799:  0.04086417685479819:
                
#                if max(i)>min(j):
#            if abs(max(i)-min(j))>0.4 and abs(min(k)-max(l))< 0.2:
            if abs(max(i)-min(j))>0.4:
#                if abs(i[0]-i[-1])<0.1:
                
                    c[leni,2]=1
                    g+=1
#        print(g,k)
        
            
#                else:
#                    c[leni,2]=1
#                    g+=1
                    
#                elif 
#        print(g)
#    
        
        
        p=0
        f=0
        
        h=0
#    print(dd) 
#    for i in dd:
         
#        if i>0.04086417685479819:
#         if i>244:
#       
#            c[p,2]=1
#            
##            print('hi')
#        p=p+1
        

        #feature extraction
        sin_list = []
        cos_list = []
        x_sp_list = []
        y_sp_list = []
        pen_up_list = []
        writing_sin = []
        writing_cos = []
        
        
        for stroke in r.findall('StrokeSet/Stroke'):
            x_point, y_point, time_list = [], [], []
            
            for point in stroke.findall('Point'):
                x_point.append(int(point.get('x')))
                y_point.append(int(point.get('y')))
                if len(time_list) == 0:
                    first_time = float(point.get('time'))
                    time_list.append(0.0)
                else:
                    time_list.append(
                                float(point.get('time')) - first_time)
            
            # calculate cos and sin
            x_point[:] = [(point - current_Xmin) * scale for point in x_point]
            y_point[:] = [(point - current_Ymin) * scale for point in x_point]
            
            

            angle_stroke = []
            if len(x_point) < 3:
#                print("Oh no",len(x_point))
                for _ in range(len(x_point)):
                    sin_list += [0]
                    cos_list += [1]
            else:
                for idx in range(1, len(x_point) - 1):
                    x_prev = x_point[idx - 1]
                    y_prev = y_point[idx - 1]
                    x_next = x_point[idx + 1]
                    y_next = y_point[idx + 1]
                    x_now = x_point[idx]
                    y_now = y_point[idx]
                    p0 = [x_prev, y_prev]
                    p1 = [x_now, y_now]
                    p2 = [x_next, y_next]
                    v0 = np.array(p0) - np.array(p1)
                    v1 = np.array(p2) - np.array(p1)
                    angle = np.math.atan2(
                                np.linalg.det([v0, v1]), np.dot(v0, v1))
                    angle_stroke.append(angle)
                new_angle_stroke = [0] + angle_stroke + [0]
                sin_stroke = np.sin(new_angle_stroke).tolist()
                cos_stroke = np.cos(new_angle_stroke).tolist()
                sin_list += sin_stroke
                cos_list += cos_stroke
                    
            # calculate speed
            if len(x_point) < 2:
                    for _ in range(len(x_point)):
                        x_sp_list += [0]
                        y_sp_list += [0]

                    if len(x_point) < 1:
                        print("Meet 0")
                        exit()
                    x_sp = [0]
                    y_sp = [0]

            else:
                    time_list = np.asarray(time_list, dtype=np.float32)
                    time_list_moved = np.array(time_list)[1:]
                    time_diff = np.subtract(
                        time_list_moved, time_list[:-1])
                    for idx, v in enumerate(time_diff):
                        if v == 0:
                            time_diff[idx] = 0.001
                    x_point_moved = np.array(x_point)[1:]
                    y_point_moved = np.array(y_point)[1:]
                    x_diff = np.subtract(x_point_moved, x_point[:-1])
                    y_diff = np.subtract(y_point_moved, y_point[:-1])
                    x_sp = np.divide(x_diff, time_diff).tolist()
                    y_sp = np.divide(y_diff, time_diff).tolist()
                    x_sp = [0] + x_sp
                    y_sp = [0] + y_sp
                    x_sp_list += x_sp
                    y_sp_list += y_sp
            # pen up and down
            pen_up = [1] * (len(x_point) - 1) + [0]
            pen_up_list += pen_up
            # writing direction
            w_sin_stroke = []
            w_cos_stroke = []
            for idx, x_v in enumerate(x_sp):
                y_v = y_sp[idx]
                slope = np.sqrt(x_v * x_v + y_v * y_v)
                if slope != 0:
                    w_sin_stroke.append(y_v / slope)
                    w_cos_stroke.append(x_v / slope)
                else:
                    w_sin_stroke.append(0)
                    w_cos_stroke.append(1)
            writing_sin += w_sin_stroke
            writing_cos += w_cos_stroke
        #end of feature extraction
        
        bigarr=np.stack((x_sp_list, y_sp_list,sin_list, cos_list,x_list, y_list, writing_sin, writing_cos,time_stamp, pen_up_list), axis=1)
        
        bigarr2=np.zeros((g,2000,10))
        hh=0
        ff=0
        for i,j in zip(c,bigarr):
            if i[2] == 0:
                bigarr2[hh,ff]=j
                ff+=1
            else:
                bigarr2[hh,ff]=j
                ff=0
                hh+=1
        
                
        
#        tri=np.zeros((g,1940,4))
#    
#        for ind,i in enumerate(c):
#        
#            if i[2]==0:
#            
#                tri[h,f,0]=i[0]
#                tri[h,f,1]=i[1]
#                tri[h,f,2]=i[3]
#                
#            
#                f=f+1
#
#            else:
#                tri[h,f,0]=i[0]
#                tri[h,f,1]=i[1]
#                tri[h,f,2]=i[3]
#                f=0
#                h=h+1
                

        

#        print(tri.shape)
#        for l,i in enumerate(tri):
#            for j,k  in enumerate(i):
#                
#                if k[0]==0 and k[1]==0:
##                    print (k)
##                    print (tri[l,j])
#                    np.delete(tri,[l,j])
#        print (tri.shape)
               
        if g==k:
            for i in bigarr2:
#                for h in i:
##          x_train=np.append(x_train,i)
##            print(i)
##           line_count =line_count+1
#                    if h[0]!=0 and h[1]!=0:
                        X.append(i)
            
#                leng=100-len(j)
#                z=np.pad(j,(0,leng),'constant')
            
            for j in words:
                
                Y.append(j)
#                n=n+1
#                with open('%i.txt'%n, 'w') as file:
#                    file.writelines("%s " % place for place in j)
#                    file.writelines("\t")
#                    for h in i:
##                        print(h[0],h[1])
#                        if h[0]!=0 and h[1]!=0:
#                            
#                            file.writelines("%s   " % place for place in h)
#                            file.writelines("\n" % place for place in h)
#                    file.close()
                            
#            with open('1.txt', 'r') as f:
#                lines=f.read()
#                y_text,x_text = lines.split('\t')
#                f.close()
#            print(y_text)
#            print('\n')
#            print(x_text)
                
                         
#                file.close()
#text_file = open("1.txt", "r")
#lines = text_file.readlines()
#print (lines)
#print (len(lines))
#text_file.close()
##                
                
#           for i,j in zip(tri, words):
##               print(j)
               
#               
#      
#        for word in tri:
#            plt.scatter(*zip(*word))
#        plt.gca().invert_yaxis() # gca= Get Current Axis
#        plt.show()
#                
#        for i in tri:
##          x_train=np.append(x_train,i)
##            print(i)
#           line_count =line_count+1
#           X.append(i)
#print(line_count)

        
#        
#        
#        
##            print (i)
##        print(y_train)
#            
Y=np.asarray(Y)
X=np.asarray(X)
Y_cat = to_categorical(Y, num_classes=None)

#print(X.shape)            
#print(Y.shape)


        

 

X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=1)
##X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
y_train2=np.zeros(y_train.shape)
for i, target_text in enumerate(y_train):
    for t, char in enumerate(target_text):
        if t > 0:
	        y_train2[i,t-1] = char


print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)            
print ('data done')
#print(y_train)

#sampled_token_indexx = 0
#for mm in y_test:
#    for rr in mm:
#        for i,val in enumerate(rr):
#            if val == 1:
#                sampled_token_indexx = i
#        if sampled_token_indexx in range (1,27):
#            sampled_token_indexx += 64
#            sampled_char = chr(sampled_token_indexx)
#        elif sampled_token_indexx in range (27,53):
#            sampled_token_indexx += 70
#            sampled_char = chr(sampled_token_indexx)
#        elif sampled_token_indexx in range (53,63):
#            sampled_token_indexx += 5
#            sampled_char = chr(sampled_token_indexx)
#        elif sampled_token_indexx in range (63,80):
#            for sym, num in symbols.items():    
#                if num == sampled_token_indexx:
#                    sampled_char = sym
#                    
#        elif sampled_token_indexx == 0:
#            sampled_char = '\t'
#        elif sampled_token_indexx == 80:	
#            sampled_char = '\n'
#        print(sampled_char)
#for stroke in X_train[0]:
#    plt.plot(*zip(*X_train))
#plt.gca().invert_yaxis() # gca= Get Current Axis
#plt.show()

            
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)
#print(X_val.shape, y_val.shape)
#print(type(Y[0]))


#            
#y_ = to_categorical(y_train)
#model = Sequential()
#model.add(LSTM(50, return_sequences=True,
#               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(Flatten())
#model.add(Dense(1, activation='softmax'))
#
#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#
#
#
#history=model.fit(X_train, y_,
#          batch_size=50, epochs=5,
#          validation_data=(X_val, y_val))
#
#score=model.evalutate(X_test,y_test, verbose=0)
##print('test loss:', score[0])
##print('test accuracy:', score[1])
#print(model.summary())
        

        
   
    
    

