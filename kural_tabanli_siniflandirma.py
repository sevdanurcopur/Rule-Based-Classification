#############################################
#Calculating Lead Yield with Rule-Based Classification
#############################################

################# Before Application #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After Application #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# TASK 1: Import Libraries
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Let's read the data set
df = pd.read_csv("datasets/persona.csv")
df.head()

# Let's see how many unique sources there are.
df["SOURCE"].nunique()
df["SOURCE"].unique()# 2
df["SOURCE"].value_counts()

# How many unique PRICEs are there?
df["PRICE"].nunique()
df["PRICE"].unique()

# How many sales were made from which PRICE?
df["PRICE"].value_counts()

# How many sales were made from which country?
df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count() 

# Let's look at how much was earned from sales in total by country?
df.groupby("COUNTRY").agg({"PRICE": ["sum","max","count"]})
df.groupby("COUNTRY")["PRICE"].sum()
df.head() 

# What are the sales numbers according to SOURCE types?
df.groupby("SOURCE").agg({"PRICE": "count"})
df["PRICE"].value_counts()
df.head()
df.groupby("SOURCE")["PRICE"].count()

# What are the PRICE averages by country?
df.groupby("COUNTRY").agg({"PRICE": "mean"})

# What are the PRICE averages according to SOURCEs?
df.groupby("SOURCE").agg({"PRICE": "mean"})

# What are the PRICE averages in the COUNTRY-SOURCE breakdown?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

#############################################
# TASK 2: What are the average earnings in the COUNTRY, SOURCE, SEX, AGE breakdown?
#############################################
selected_col = ["COUNTRY", "SOURCE", "SEX", "AGE"]
new_df = df.groupby(selected_col).agg({"PRICE": "mean"})

#############################################
# GÖREV 3: Let's sort the output by PRICE
#############################################
agg_df = new_df.sort_values(by=['PRICE'], ascending=False)

#############################################
# TASK 4: Let's convert the names in the index into variable names.
#############################################
agg_df.reset_index(inplace=True)

#############################################
# GÖREV 5: Lets Convert the AGE variable to a categorical variable and add it to agg_df.
#############################################

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70],
                           labels=['0_18', '19_23', '24_30', '31_40', '41_70'])

mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
pd.qcut(agg_df['AGE'], q=5, labels=mylabels)

#############################################
# Task 6:Define new level based customers and add them to the data set as a variable.
#############################################

agg_df["customer_level_based"] = [(country + "_" + source + "_" + sex + "_" + age_cat).upper() for country, source, sex, age_cat in zip(agg_df["COUNTRY"], agg_df["SOURCE"], agg_df["SEX"], agg_df["AGE_CAT"])]

#agg_df["customers_level_based"] = ["_".join(map(str, a)).upper() for a in agg_df.drop(["AGE", "PRICE"], axis=1).values]
agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df.apply(lambda x: x.astype(str).str.upper())

agg_df = agg_df.groupby("customer_level_based").agg({"PRICE": "mean"})
agg_df.reset_index(inplace=True)

#############################################
# TASK 7: Let's Segment new customers 
#############################################
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

#############################################
# TASK 8: Let's Classify new customers and estimate how much income they can bring.
#############################################
# To which segment does a 33-year-old Turkish woman using ANDROID belong and how much income is she expected to earn on average?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == new_user]

# In which segment and how much income on average is a 35-year-old French woman using IOS expected to earn?
new_user_2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == new_user_2]

