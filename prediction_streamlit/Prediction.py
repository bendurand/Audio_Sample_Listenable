#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
from tqdm import tqdm
from tensorflow import keras
import streamlit as st


def save_uploadedfile(uploadedfile): #Fonction appelée pour s'assurer du bon format du fichier audio
    with open(os.path.join("prediction_streamlit/Sound_File/", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to Data".format(uploadedfile.name))

def main():
    model = keras.models.load_model('./models/2023-01-11-22-59-32') #Charger le modèle keras
    feature_list = [] #Création des listes utilisées pour la prediction
    Name_list=[]

    st.title('Bienvenue sur l\'application prediction !')
    st.text("Ici vous pourrez uploader des sons sous format .wav qui seront directement notés")
    st.text(" par notre algorithme.")

    Files=st.file_uploader("Uploadez vos ficher en .wav :",accept_multiple_files=True) #Création du système de drop&drag
    
    if Files is not None: #fonction pour s'assurer qu'il y ait bien des fichiers selectionnées pour l'utilisateur
        for file in Files:
            save_uploadedfile(file)

    else:
        st.text('You have not uploaded any files yet')            


    print('Preparing feature dataset and labels.')
    for file in tqdm(os.listdir('./Sound_File')): #iteration à travers les fichiers dans le dossier sound_file
        # Skip si ce n'est pas un fichier wav
        if not file.endswith('.wav'): 
            continue
        # charge l'audio et l'étend à 1 seconde
        audio_path = os.path.join('./Sound_File/', file)
        
        audio, sr = librosa.load(path=audio_path, sr=None)
        audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr)
        # Calculate features and get the label from the filename
        mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512) #création du melspectrogram
        mels_db = librosa.power_to_db(S=mels, ref=1.0)
        if mels_db.shape == (128,87): #on s'assure que chaque feature a les mêmes dimensions
            feature_list.append(mels_db.reshape((128, 87, 1)))
            Name_list.append(file)

    if(len(feature_list)!=0): #on lance la prediction si il y a des éléments à prédire
        features = np.array(feature_list) 
        x=model.predict(features) #commande de prediction : renvoie un nombre entre 0 et 1 pour chaques sons
        for i in range(len(x)):
        
            if(x[i][0]>0.5): #si note > 0.5 : son écoutable et vice versa
                st.text(body=('le son ',Name_list[i],' est écoutable'))
                print(i," : ", "son écoutable")
            else:
                st.text(body=('le son ',Name_list[i],' n\'est pas écoutable'))
                print(i," : ", "son non écoutable") 

    else:
        st.write("Aucun fichier selectionné pour le moment")



    
    if st.button('Supprimer les fichier'): #bouton pour supprimer les fichiers dans le dossier sound file
        for file in tqdm(os.listdir('./Sound_File')):
            if file == None:
                continue
            audio_path = os.path.join('./Sound_File/', file)
            os.remove(audio_path)

    
    



    

if __name__ == "__main__":
    main()