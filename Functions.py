#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# When using this code please refer to this work: "Guglielmo, G., Klincewicz, M., Huis in 't Veld, E., Spronck, P. Tracking Early Differences in Tetris Performance using Eye Aspect Ratio Extracted Blinks. (2023). IEEE Transactions on Games"



import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from sklearn.ensemble import IsolationForest
import more_itertools as mit
from scipy import stats



# this first function is mostly based on the isolation forest authored by  ARJUN SRIDHAR, ASHEN FERNANDO
# you can find the original code at this link: https://github.com/mi3nts/tobiiBlinkDetection/tree/main/Isolation%20Forest


# if you need a code to extract the EAR please look at the example in the notebook file

# the values of 2, 5, 15 represent constraints of frames representing milliseconds (2*33, 2*33, 15*33) in a 30 fps recording

# the values of 2, 4, 15 represent constraints of frames representing milliseconds (2*33, 4*33, 15*33) in a 30 fps recording

# 1 frame = 33 ms (approximately)

def isolation_forest(data):

    # number of standard deviations away from the rolling median 
    devs = 2.5

    # size of the rolling window
    roll_window = 100



    # pot_outliers will contain points below 2.5 sigma away from rolling EAR_Avg 
    

    rolling = data['EAR_Avg'].rolling(roll_window).median()
    rolling_abs = rolling - devs*stats.median_abs_deviation(rolling,nan_policy = "omit")
    


    

    pot_outliers = data.loc[data['EAR_Avg'] < rolling_abs]

    # a first order estimation of contamination, a ratio of data 2.5 sigma away from absolute median to total data
    contam = len(pot_outliers)/len(data)

    # implement isolation forest, contamination optimized using the median
    data_np = data['EAR_Avg'].to_numpy().reshape(-1,1)

    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=contam, random_state=1234)

    fit = model.fit(data_np)
    decision = model.decision_function(data_np)
    pred = model.predict(data_np)

    # separate outliers (with a score of -1) from normal samples

    isf = pd.DataFrame({'dec':decision, 'pred':pred})

    ears = pd.DataFrame({'inds':isf.loc[isf['pred'] == -1].index, 'EAR_vals':data['EAR_Avg'][isf.loc[isf['pred'] == -1].index]})
    ears = ears[ears['EAR_vals'] < ears['EAR_vals'].mean()]

    # creates a list of lists that keeps track of groups of consecutive records
    blinks_list_iso = [list(group) for group in mit.consecutive_groups(ears.index)]

    # counts the number of blinks and where they occur, given there are consecutive records (i.e. duration of the predicted blink) 
    # is longer than metric specified by dur
    count = 0
    blinks_iso_grouped = []
    
    for i in blinks_list_iso:
        if len(i) >= 2 and len(i) <=  15: # checks for number of frames long between 50 ms and 500 ms (approx to 66 ms and 500 ms with 30 fps)
            blinks_iso_grouped.append(i)
            count += 1
    
    # flatten the grouped list, to be used for validation 
    flat_list = [item for sublist in blinks_iso_grouped for item in sublist]

    # return a dataframe/csv with with 'Frame', 'EAR_Avg', 'Classification'
    data_dict = {'Frame': np.arange(0,len(data)), 'EAR_Avg': data['EAR_Avg'], 'Classification': np.zeros(len(data), dtype='int')}

    data_df = pd.DataFrame.from_dict(data_dict)

    # index into df using flat list (which has correct blink flags) to set classification value to true  (1 is assigned to frames belonging to blinks)
    data_df['Classification'].loc[data_df.index[flat_list]] = 1

    #data_df.to_csv("classification_frames.csv", index=False)
    return(data_df)







def consecutive(data, stepsize=1): # expect list of frames to be already in order since the frames are given in sorted order (starting from 0)
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)



def interval_checker(file): # check consecutives and makes sure that there are at least 4 frames of distance between one blink and another
    test2 = consecutive(file)
    #test2 = consecutive(res["Frame"][res["Classification"] == 1].tolist())

    selected_end = list() 
    selected_beg = list()
    to_fill = list()
    unpacked_numbers = list()
    unpacked_missing = list()

    for i in test2:
        selected_end.append(i[-1]) # select the end of each set of ordered frames, 
        selected_beg.append(i[0]) #select the beginning of each set of ordered frames
        # selected_endarr and selected begarr define the two extremities used to fill in the missing data 
        selected_endarr = selected_end[:-1] #first and last points are not of interest. they are used to contain the list with itself 
        selected_begarr = selected_beg[1:] 
        result = [None]*(len(selected_begarr)+len(selected_endarr)) # allocate free space list as long as the 2 used lists
        result[::2] = selected_begarr # this is to make sure that the smallest number comes first, it used to create couple between following and previous frames defined as end and beginning part of a blink and it is used to check if they respect the interval of 4 frames later on
        result[1::2] = selected_endarr 
        zipped = [list(t) for t in zip(result[::2], result[1::2])] #this part puts together each number with its closer frame to later evaluate if it respects the interval
        for j in i:
            unpacked_numbers.append(j) # unpacks the already defined consecutive frames 



    for k in zipped: # finds where the interval is lower that expected and fill-in the missing frames, generally blinks have around 100 ms of minimal interval according to previous studies
        start = k[-1]
        end = k[0]
        sub = end-start
        if sub < 4: # this is the minimal interval betwen blinks which is expected to be at least 100 ms for this reason everything below 4 frames (132 ms) is filled in as outlier
            to_fill.append(k)


    for x in to_fill: #fills the 2 closest frames of each blink where the conditio of <4 is not respected
        for w in (list((range(x[1],x[0])))):
            unpacked_missing.append(w) 


    unique = sorted(list(set(unpacked_missing + unpacked_numbers))) # final list containing all the frames that should be considered as part of blinks
    # the use of the set commands prevents the existence of doubles and returns the sorted frames in order 

    return(unique)




def duration_blinks(listrand): # this function extracts the length of frames beloning to blinks and groups them, it is applied as double check after having controlled for the intervals (intervals_checker function)
    count=1
    consec_list=[]
    durations = list()
    for i in range(len(listrand[:-1])):
        if listrand[i]+1 == listrand[i+1]:
            count+=1
        else:
            consec_list.append(count)
            count=1

    # Account for the last iteration
    consec_list.append(count)
    for j in consec_list:
        if j >= 2 and j <= 15: # range approx between 50 ms and 500 ms with 30 fps recording
            durations.append(j)

    return durations


def blink_interval_extr(all_frames, blink_frames): # this function extracts the frames not belonging to blinks
    interval_to_clean = list()
    for i in all_frames: 
        if i not in blink_frames: 
            interval_to_clean.append(i)

    interval_to_clean= sorted(interval_to_clean)
    return(interval_to_clean)

def blink_frames_finder(lista): # this function is used to finalize the frames beloning to blinks. it is used to filter all the frames found by the isolation forest to find the intervals
    final_frames = list()
    for i in consecutive(lista):
         if len(i) >= 2 and len(i) <= 15: # range approx between 50 ms and 500 ms with 30 fps recording. appends just sets of frames representing blinks with a specific duration between approx 50 and 500 frames
                final_frames.append(i)
    flatList = [element for innerList in final_frames for element in innerList]
    return(flatList)

