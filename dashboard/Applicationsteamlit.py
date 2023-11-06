import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Vérification de la solvabilité et de l'âge")

# Chargement des données
data_path = "..data\\application_train.csv"
data = pd.read_csv(data_path)
st.write("Le fichier a été lu avec succès!")

# Prétraitement des données
data['AGE'] = data['DAYS_BIRTH'] / -365
bins = np.linspace(20, 70, num=11)
data['AGE_BIN'] = pd.cut(data['AGE'], bins=bins)
st.write("Prétraitement des données terminé avec succès!")

# Histogramme des groupes d'âge
age_groups = data.groupby('AGE_BIN')['TARGET'].mean()
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.bar(age_groups.index.astype(str), 100 * age_groups)

plt.xticks(rotation=75)
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')
plt.tight_layout()
st.pyplot(fig1)

# Diagramme circulaire de la solvabilité
target_distribution = data['TARGET'].value_counts()
labels = ['Solvable (TARGET=0)', 'Non-Solvable (TARGET=1)']
colors = ['#66b2ff', '#ff9999']
explode = (0.1, 0) 
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.pie(target_distribution, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribution of Solvability')
plt.tight_layout()
st.pyplot(fig2)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load train data
train_data = pd.read_csv("..\\data\\application_train.csv")

# Load test data
test_data = pd.read_csv("..\\data\\application_test.csv")
st.write("Fichier `application_test.csv` chargé avec succès!")
st.write(test_data.head())

# Calculate age from DAYS_BIRTH for train data
train_data['AGE'] = train_data['DAYS_BIRTH'] / -365

# Extracting client 100001 data from train data
client_data = train_data[train_data['SK_ID_CURR'] == 100002]

# Displaying client 100001 data
st.title("Informations sur le client 100002")
st.write(client_data[['AGE', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN']])

# Histograms
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Age histogram
ages = train_data['AGE']
axes[0].hist(ages, bins=20, color='blue', alpha=0.7, label='Tous les clients')
axes[0].axvline(client_data['AGE'].values[0], color='red', linestyle='dashed', linewidth=2, label='Client 100002')
axes[0].set_title("Distribution des âges")
axes[0].set_xlabel("Âge")
axes[0].set_ylabel("Nombre de clients")
axes[0].legend()

# Marital status histogram
statuses = train_data['NAME_FAMILY_STATUS'].value_counts()
statuses.plot(kind='bar', ax=axes[1], color='blue', alpha=0.7, label='Tous les clients')
axes[1].set_title("Répartition du statut marital")
axes[1].set_ylabel("Nombre de clients")
status_client = client_data['NAME_FAMILY_STATUS'].values[0]
axes[1].bar(status_client, statuses[status_client], color='red', label='Client 100002')
axes[1].legend()

# Number of children histogram
children_counts = train_data['CNT_CHILDREN'].value_counts()
children_counts.plot(kind='bar', ax=axes[2], color='blue', alpha=0.7, label='Tous les clients')
axes[2].set_title("Répartition du nombre d'enfants")
axes[2].set_xlabel("Nombre d'enfants")
axes[2].set_ylabel("Nombre de clients")
children_client = client_data['CNT_CHILDREN'].values[0]
axes[2].bar(children_client, children_counts[children_client], color='red', label='Client 100002')
axes[2].legend()

st.pyplot(fig)

# Pie chart for solvency
solvent = train_data[train_data['TARGET'] == 0].shape[0]
insolvent = train_data[train_data['TARGET'] == 1].shape[0]
labels = ['Solvable', 'Insolvable']
sizes = [solvent, insolvent]
colors = ['blue', 'red']
explode = (0, 0.1) if client_data['TARGET'].values[0] == 1 else (0.1, 0)
fig2, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')
st.title("Solvabilité des clients")
st.pyplot(fig2)

st.write(f"Nombre total de clients (données d'entraînement) : {train_data.shape[0]}")
