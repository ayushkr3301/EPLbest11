# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 09:48:51 2019

@author: hp
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
from scipy.stats import norm
from operator import itemgetter


# Importing the dataset
dataset = pd.read_csv('data.csv')

dataset['Wage'] = dataset['Wage'].map(lambda x: str(x)[:-1])
dataset['Wage'] = dataset['Wage'].map(lambda x: str(x)[1:])
dataset['Wage'] = pd.to_numeric(dataset['Wage'])

dataset['Value'] = dataset['Value'].map(lambda x: str(x)[1:])        
dataset['Release Clause'] = dataset['Release Clause'].map(lambda x: str(x)[1:])

dataset = dataset.drop([ "Photo", "Flag", "Club Logo", "Body Type", "Special",
                        "Real Face", "Joined", "Loaned From", "LS",  "ST", "RS",
                        "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                        "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM",
                        "RWB", "LB", "LCB", "CB", "RCB", "RB", "Unnamed: 0",
                        "ID" ], axis = 1)

dataset = dataset.dropna()

dataset.sort_values("Name", inplace=True) 


#Create a dataset of players of prominent English clubs

epl = dataset[(dataset['Club'] == 'Manchester City') | (dataset['Club'] == 'Liverpool')
                | (dataset['Club'] == 'Chelsea') | (dataset['Club'] == 'Arsenal') 
                | (dataset['Club'] == 'Tottenham Hotspur') | 
                (dataset['Club'] == 'Manchester United') | (dataset['Club'] == 'Everton')
                | (dataset['Club'] == 'Wolverhampton Wanderers') | (dataset['Club'] ==  'Leicester City')
                | (dataset['Club'] == 'Southampton') | (dataset['Club'] == 'Watford')
                | (dataset['Club'] == 'Burnley') | (dataset['Club'] == 'Crystal Palace')
                | (dataset['Club'] == 'Newcastle United') | (dataset['Club'] == 'West Ham United')]
                



#Create the best 15

#GOALKEEPERS

#weights
a = 0.5
b = 1
c= 2
d = 3

epl['gk_Sweeper'] = ((b * epl['Reactions']) + (b * epl['Composure']) + (b * epl['Acceleration']) + (a * epl['ShortPassing']) + (a * epl['LongPassing']) + (b * epl['Jumping']) + (b * epl['GKPositioning']) + (b * epl['GKDiving']) + (d * epl['GKReflexes']) + (b * epl['GKHandling']) + (d * epl['GKKicking'] ))/(14)

plt.figure(figsize=(15,6))
sd = epl.sort_values('gk_Sweeper', ascending=False)[:5]
x2 = np.array(list(sd['Name']))
y2 = np.array(list(sd['gk_Sweeper']))
sns.barplot(x2, y2, palette= "colorblind")
plt.ylabel("Sweeping Score")


#CENTRAL DEFENDERS

epl['df_centre_backs'] = ( d*epl['Reactions'] + c*epl['Interceptions'] + d*epl['SlidingTackle'] + d*epl['StandingTackle'] + b*epl['Composure']  +a*epl['ShortPassing'] + b*epl['LongPassing']+ c*epl['Acceleration'] + d*epl['Stamina'] + d*epl['Jumping'] + d*epl['HeadingAccuracy'] +  d*epl['Marking'] + c*epl['Aggression'])/(29.5)


plt.figure(figsize=(15,6))
sd = epl[(epl['Position'] == 'LCB')].sort_values('df_centre_backs', ascending=False)[:5]
x2 = np.array(list(sd['Name']))
y2 = np.array(list(sd['df_centre_backs']))
sns.barplot(x2, y2, palette=sns.color_palette("Blues_d"))
plt.ylabel("LCB Score")
    

plt.figure(figsize=(15,6))
sd = epl[(epl['Position'] == 'RCB')].sort_values('df_centre_backs', ascending=False)[:5]
x2 = np.array(list(sd['Name']))
y2 = np.array(list(sd['df_centre_backs']))
sns.barplot(x2, y2, palette=sns.color_palette("Blues_d"))
plt.ylabel("RCB Score")


#FULL BACKS

epl['df_wb_Wing_Backs'] = (b*epl['BallControl'] + a*epl['Dribbling'] + a*epl['Marking'] + d*epl['SlidingTackle'] + d*epl['StandingTackle'] + a*epl['Positioning']  + d*epl['Crossing'] + b*epl['ShortPassing'] + c*epl['LongPassing'] + d*epl['Acceleration'] +d*epl['SprintSpeed'] + c*epl['Stamina'])/(22.5)


plt.figure(figsize=(15,6))
 
sd = epl[(epl['Position'] == 'LWB') | (epl['Position'] == 'LB')].sort_values('df_wb_Wing_Backs', ascending=False)[:5]
x4 = np.array(list(sd['Name']))
y4 = np.array(list(sd['df_wb_Wing_Backs']))
sns.barplot(x4, y4, palette=sns.color_palette("Blues_d"))
plt.ylabel("Left Back Score")


plt.figure(figsize=(15,6))

sd = epl[(epl['Position'] == 'RWB') | (epl['Position'] == 'RB')].sort_values('df_wb_Wing_Backs', ascending=False)[:5]
x5 = np.array(list(sd['Name']))
y5 = np.array(list(sd['df_wb_Wing_Backs']))
sns.barplot(x5, y5, palette=sns.color_palette("Blues_d"))
plt.ylabel("Right Back Score")


#PLAYMAKER

epl['mf_playmaker'] = (d*epl['BallControl'] + d*epl['Dribbling'] + a*epl['Marking'] + d*epl['Reactions'] + d*epl['Vision'] + c*epl['Positioning'] + c*epl['Crossing'] + d*epl['ShortPassing'] + c*epl['LongPassing'] + c*epl['Curve'] + b*epl['LongShots'] )/(24.5)


plt.figure(figsize=(15,6))
 
ss = epl[(epl['Position'] == 'CAM') | (epl['Position'] == 'LAM') | (epl['Position'] == 'RAM')].sort_values('mf_playmaker', ascending=False)[:5]
x3 = np.array(list(ss['Name']))
y3 = np.array(list(ss['mf_playmaker']))
sns.barplot(x3, y3, palette=sns.diverging_palette(145, 280, s=85, l=25, n=5))
plt.ylabel("PlayMaker Score")


#CONTROLLER

epl['mf_controller'] = (b*epl['Weak Foot'] + d*epl['BallControl'] + b*epl['Dribbling'] + a*epl['Marking'] + b*epl['Reactions'] + c*epl['Vision'] + d*epl['Composure'] + d*epl['ShortPassing'] + d*epl['LongPassing'])/(17.5)


plt.figure(figsize=(15,6))
 
# Generate some sequential data
ss = epl[(epl['Position'] == 'LCM') | (epl['Position'] == 'RCM') | (epl['Position'] == 'CM')].sort_values('mf_controller', ascending=False)[:5]
x1 = np.array(list(ss['Name']))
y1 = np.array(list(ss['mf_controller']))
sns.barplot(x1, y1, palette=sns.diverging_palette(145, 280, s=85, l=25, n=5))
plt.ylabel("Controller Score")


#MIDFIELD-BEAST

epl['mf_beast'] = (c*epl['Balance'] + c*epl['Jumping'] + c*epl['Strength'] + d*epl['Stamina'] + a*epl['SprintSpeed'] + c*epl['Acceleration'] + d*epl['ShortPassing'] + c*epl['Aggression'] + d*epl['Reactions'] + c*epl['Marking'] + d*epl['StandingTackle'] + d*epl['SlidingTackle'] + d*epl['Interceptions'])/(30.5)


plt.figure(figsize=(15,6))
 
ss = epl[(epl['Position'] == 'RDM') | (epl['Position'] == 'LDM') | (epl['Position'] == 'CDM')].sort_values('mf_beast', ascending=False)[:5]
x2 = np.array(list(ss['Name']))
y2 = np.array(list(ss['mf_beast']))
sns.barplot(x2, y2, palette=sns.diverging_palette(145, 280, s=85, l=25, n=5))
plt.ylabel("Beast Score")


#WINGERS


epl['att_wing'] = (c*epl['Weak Foot'] + d*epl['BallControl'] + d*epl['Dribbling'] + d*epl['SprintSpeed'] + d*epl['Acceleration'] + b*epl['Vision'] + c*epl['Crossing'] + b*epl['ShortPassing'] + b*epl['LongPassing'] + c*epl['Agility'] + b*epl['Curve'] + b*epl['FKAccuracy'] + d*epl['Finishing'])/(26)


plt.figure(figsize=(15,6))
 
ss = epl[(epl['Position'] == 'LW') | (epl['Position'] == 'LM') | (epl['Position'] == 'LS')].sort_values('att_wing', ascending=False)[:5]
x1 = np.array(list(ss['Name']))
y1 = np.array(list(ss['att_wing']))
sns.barplot(x1, y1, palette=sns.diverging_palette(255, 133, l=60, n=5, center="dark"))
plt.ylabel("Left Wing")




plt.figure(figsize=(15,6))
 
ss = epl[(epl['Position'] == 'RW') | (epl['Position'] == 'RM') | (epl['Position'] == 'RS')].sort_values('att_wing', ascending=False)[:5]
x2 = np.array(list(ss['Name']))
y2 = np.array(list(ss['att_wing']))
sns.barplot(x2, y2, palette=sns.diverging_palette(255, 133, l=60, n=5, center="dark"))
plt.ylabel("Right Wing")



#STRIKER

epl['att_striker'] = (b*epl['Weak Foot'] + b*epl['BallControl'] + b*epl['Aggression'] + b*epl['Acceleration'] + a*epl['Curve'] + c*epl['LongShots']+ c*epl['ShotPower'] + c*epl['Balance'] + d*epl['Finishing'] + d*epl['HeadingAccuracy'] + c*epl['Jumping'] + c*epl['Dribbling'])/(20.5)


plt.figure(figsize=(15,6))
ss = epl[(epl['Position'] == 'ST') | (epl['Position'] == 'LS') | (epl['Position'] == 'RS') | (epl['Position'] == 'CF')].sort_values('att_striker', ascending=False)[:5]
x3 = np.array(list(ss['Name']))
y3 = np.array(list(ss['att_striker']))
sns.barplot(x3, y3, palette=sns.diverging_palette(255, 133, l=60, n=5, center="dark"))
plt.ylabel("Striker")


