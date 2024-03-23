#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().system('pip install basemap')
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
from matplotlib import animation,rc
import io
import base64
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
# from scipy.misc import imread
import codecs
from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# # 1.1 Gathering Data

# In[6]:


df = pd.read_csv("C:\\Users\\suhai\\Downloads\\Global Terrorism - START data\\globalterrorismdb_0718dist.csv", encoding = 'ISO-8859-1')


# In[7]:


df.head()


# In[8]:


df.columns


# In[9]:


df.shape


# In[10]:


df.describe()


# # 1.2 Data Preprocessing 

# In[11]:


# Renaming Columns 

df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',
                     'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                     'nwound': 'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                     'weaptype1_txt':'Weapon_type','motive':'Motive'
                    },inplace=True)


# In[12]:


# As we have many columns, we take the columns that are necessary for analysis
df = df[['eventid','Year','Month','Day','Country','Region','state','city','latitude','longitude','AttackType','Killed','Wounded',
            'Target','Summary','Group','Target_type','Weapon_type','Motive','success']]


# In[13]:


df['Killed'].sample(10)


# **Create a new column 'casualties' by adding 'killed' and 'wounded'**

# In[14]:


df['casualities']=df['Killed']+df['Wounded']
df.head(3)


# In[15]:


df.shape


# In[16]:


df.isna().sum()


# In[17]:


df.describe()


# # 2.0 Exploratory Data Analysis

#  ## Number Of Terrorist Acticity Each Years

# In[18]:


year_attacks = df.groupby('Year').size().reset_index(name='count')
sns.lineplot(x='Year', y='count', data=year_attacks, color='red')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.title("Number of Terrorist Acticity")
plt.show()


# In[19]:


plt.subplots(figsize=(15,6))
sns.countplot(data=df, x='Year', palette='inferno')
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# ***There has been a steady increase in global terrorist activities year by year. However, the year 2014 stands out as the peak with the highest recorded incidents. Encouragingly, there has been a subsequent decline in terrorist activity post-2014, offering hope for improved global security efforts.***

# ## Terrorist Attacks Trends in Regions

# In[20]:


year_attacks_region = df.groupby(['Year','Region']).size().reset_index(name='count')


# In[21]:


plt.subplots(figsize=(15,6))
sns.lineplot(x='Year',y='count',hue='Region',data=year_attacks_region)
plt.title('Terrorist Attacks Trends in Regions')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.show()


# In[22]:


plt.subplots(figsize=(15,6))
sns.countplot(x='Region',data=df,palette='magma',order=df['Region'].value_counts().index)
plt.xticks(rotation=80)
plt.title('Number Of Terrorist Activities By Region')
plt.show()


# ***Terrorism in the Middle East has experienced repeated increases year after year, largely due to ongoing geopolitical conflicts and the presence of extremist groups. South Asia has also witnessed a rise in terrorism, often linked to criminal organizations and drug trafficking. In contrast, Central Asia has comparatively lower terrorism rates***

# ## Top 10 Affected Countries

# In[23]:


plt.subplots(figsize=(12,6))
top=df['Country'].value_counts()[:10].to_frame().reset_index()
top.columns= ['Country','Attacks_Counts']
sns.barplot(x='Country',y='Attacks_Counts', data= top, palette='magma')
plt.title('Top Countries Affected')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation=80)
plt.show()


# **The graph highlights five countries most affected by terrorism:**
# 
# 1. Iraq
# 2. Pakistan
# 3. Afghanistan
# 4. India 
# 5. Colombia
# 
# **These nations face significant challenges related to terrorism, requiring ongoing efforts to ensure the safety and security of their populations and regional stability.**

# In[24]:


plt.subplots(figsize=(12,6))
top=df['city'].value_counts()[:20].to_frame().reset_index()
top.columns= ['city','Attacks_Counts']
sns.barplot(x='city',y='Attacks_Counts', data= top, palette='magma')
plt.title('Top city Affected')
plt.xlabel('city')
plt.ylabel('Count')
plt.xticks(rotation=80)


# ## Attacking Methods by Terrorists

# In[25]:


plt.subplots(figsize=(15,6))
sns.countplot(x='AttackType',data=df,palette='inferno',order=df['AttackType'].value_counts().index)
plt.xticks(rotation=80)
plt.title('Attacking Methods by Terrorists')
plt.show()


# ## Top Terrorists Group

# In[26]:


group_counts = df['Group'].value_counts()
sort = group_counts.sort_values(ascending=False)

# Select the top 5 most frequent groups
sort = sort.iloc[1:]
top_5 = sort.head(10)


# In[27]:


# Plotting top 5 terrorists groups

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(x=top_5.index, y=top_5.values, palette="magma")
plt.title('Top Terrorist Attack Groups')
plt.xlabel('Attack Count')
plt.ylabel('Group')
plt.xticks(rotation = 80)
plt.show()


# **The Taliban is a prominent terrorist group, but it's important to note that the global terrorism landscape is complex. Other significant terrorist groups, like ISIS, Al-Qaeda, Boko Haram, and Al-Shabaab, also operate in various regions, making it challenging to definitively label one as the "most active" worldwide. The prominence of these groups can change over time.**

# ## Activity of Top Terrorist Groups

# In[28]:


top_groups10=df[df['Group'].isin(df['Group'].value_counts()[1:11].index)]
pd.crosstab(top_groups10.Year,top_groups10.Group).plot(color=sns.color_palette('Paired',10))
fig=plt.gcf()
fig.set_size_inches(16,6)
plt.show()


# ## Regions Attacked By Terrorist Groups

# In[29]:


top_groups=df[df['Group'].isin(df['Group'].value_counts()[:14].index)]
m4 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
m4.drawcoastlines()
m4.drawcountries()
m4.fillcontinents(lake_color='#fff')
m4.drawmapboundary(fill_color='#fff')
fig=plt.gcf()
fig.set_size_inches(22,10)
colors=['r','g','b','y','#800000','#ff1100','#8202fa','#20fad9','#ff5733','#fa02c6',"#f99504",'#b3b6b7','#8e44ad','#1a2b3c']
group=list(top_groups['Group'].unique())
def group_point(group,color,label):
    lat_group=list(top_groups[top_groups['Group']==group].latitude)
    long_group=list(top_groups[top_groups['Group']==group].longitude)
    x_group,y_group=m4(long_group,lat_group)
    m4.plot(x_group,y_group,'go',markersize=3,color=j,label=i)
for i,j in zip(group,colors):
    group_point(i,j,i)
legend=plt.legend(loc='lower left',frameon=True,prop={'size':10})
frame=legend.get_frame()
frame.set_facecolor('white')
plt.title('Regional Activities of Terrorist Groups')
plt.show()


# # People Killed and Wounded In Each Year

# In[30]:


k=df[["Year","Killed"]].groupby("Year").sum()


# In[31]:


plt.figure(figsize=(10, 6))
sns.barplot(x=k.index, y="Killed", palette="inferno",data=k)

plt.title("Total Killings by Year")
plt.xlabel("Year")
plt.ylabel("Total Killings")

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# In[32]:


k=df[["Year","Wounded"]].groupby("Year").sum()


# In[33]:


plt.figure(figsize=(10, 6))
sns.barplot(x=k.index, y="Wounded", palette="inferno",data=k)

plt.title("Total Wounded by Year")
plt.xlabel("Year")
plt.ylabel("Total Wounded")

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ## People Killed and Wounded In Each Region

# In[34]:


k=df[["Region","Killed"]].groupby("Region").sum().sort_values(by="Killed",ascending=False)
k


# In[35]:


w=df[["Region","Wounded"]].groupby("Region").sum().sort_values(by="Wounded",ascending=False)
w


# In[36]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#People Killed
k.plot(kind="bar",color="indigo",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each Region")
ax0.set_xlabel("Regions")
ax0.set_ylabel("Number of People Killed")

#People Wounded
w.plot(kind="bar",color="green",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded in each Region")
ax1.set_xlabel("Regions")
ax1.set_ylabel("Number of People Wounded")

plt.show


# ## Types of terrorist attacks that cause deaths

# In[37]:


killData = df.loc[:,'Killed']
print('Number of people killed by terror attack:', int(sum(killData.dropna())))# drop the NaN values


# In[38]:


attackData = df.loc[:,'AttackType']
typeKillData = pd.concat([attackData, killData], axis=1)


# In[39]:


typeKillFormatData = typeKillData.pivot_table(columns='AttackType', values='Killed', aggfunc='sum')
typeKillFormatData


# In[40]:



labels = typeKillFormatData.columns.tolist() # convert line to list
transpoze = typeKillFormatData.T # transpoze

# Assuming values is a 2D array
values = transpoze.values.tolist()
values = np.array(values).flatten()  # Flatten the 2D array to make it 1D

fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(aspect="equal"))
plt.pie(values, startangle=90, autopct='%.2f%%')
plt.title('Types of terrorist attacks that cause deaths')
plt.legend(labels, loc='upper right', bbox_to_anchor=(1.3, 0.9), fontsize=15)  # location legend
plt.show()


# **The combination of armed assaults and bombings/explosions is responsible for a significant 77% of fatalities in terrorist attacks. This highlights the persistent use of these tactics and underscores the global threat posed by weapons and explosives.**

# ## Yearly Casualities

# In[41]:


plt.subplots(figsize=(15,6))
year_cas = df.groupby('Year').casualities.sum().to_frame().reset_index()
year_cas.columns = ['Year','casualities']
sns.barplot(x=year_cas.Year, y=year_cas.casualities, palette='magma',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=90)
plt.title('Number Of Casualities Each Year')
plt.show()


# ## Number of Total Casualities in Each Country

# In[42]:


plt.subplots(figsize=(15,6))
count_cas = df.groupby('Country').casualities.sum().to_frame().reset_index().sort_values('casualities', ascending=False)[:15]
sns.barplot(x=count_cas.Country, y=count_cas.casualities, palette= 'magma',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=30)
plt.title('Number of Total Casualities in Each Country')
plt.show()


# # Terrorist Attacks in India

# In[43]:


india_data = df[df['Country'] == 'India']

# Get the top 14 terrorist groups in India
top_groups = india_data['Group'].value_counts().head(14).index

# Create a Basemap instance
m4 = Basemap(
    projection='mill',
    llcrnrlat=-10,
    urcrnrlat=40,
    llcrnrlon=70,
    urcrnrlon=100,
    resolution='c',
    lat_0=True,
    lat_1=True
)

# Customize the map
m4.drawcoastlines()
m4.drawcountries()
m4.fillcontinents(lake_color='#fff')
m4.drawmapboundary(fill_color='#fff')

# Set the figure size
fig = plt.gcf()
fig.set_size_inches(22, 10)

# Define colors for plotting
colors = ['r', 'g', 'b', 'y', '#800000', '#ff1100', '#8202fa', '#20fad9', '#ff5733', '#fa02c6', "#f99504", '#b3b6b7', '#8e44ad', '#1a2b3c']

# Iterate through the top groups and plot their activities
for group, color in zip(top_groups, colors):
    group_data = india_data[india_data['Group'] == group]
    x_group, y_group = m4(group_data['longitude'].values, group_data['latitude'].values)
    m4.plot(x_group, y_group, 'go', markersize=3, color=color, label=group)

# Add legend
plt.legend(loc='lower left', frameon=True, prop={'size': 10})

# Set the plot title
plt.title('Regional Activities of Top Terrorist Groups in India')

# Show the plot
plt.show()


# In[44]:


India = df[(df['Country'] == 'India')]
India.head(5)


# In[45]:


India_attacks = India['eventid'].count()
print('There were',India_attacks ,'attacks in India.')


# In[46]:


India_success = India.groupby('success').size().reset_index(name='count')
India_success['percentage'] = India_success['count'] / India_attacks * 100
India_success


# In[47]:


sns.barplot(x='success', y='percentage', data = India_success,palette=['green', 'red'])
plt.title("Outcome of Terrorist Attacks in India")
plt.xlabel("Outcome")


# ## Attack types in India and their success rates.

# In[48]:


attack_types_India = India.groupby(['AttackType','success']).size().reset_index(name='count')
attack_types_India


# In[49]:


plt.figure(figsize=(20,10))
sns.barplot(x='AttackType', y='count', hue='success', data=attack_types_India,  palette= 'magma')
plt.xticks(rotation=30)
plt.title("Facility ")


# In[50]:


nkills_India = India.groupby('AttackType')[['Killed']].sum().reset_index()
nkills_India


# In[51]:


plt.figure(figsize=(30,10))
sns.barplot(x='AttackType', y='Killed', data=nkills_India,palette= 'magma')


# # Conclusion
# 
# **The global landscape is witnessing a concerning rise in the incidence of terrorism attacks, posing a growing threat to peace and security. This unsettling trend is particularly pronounced in two regions: the Middle East and North Africa, as well as South America, where the number of terrorist attacks has surged significantly.**
# 
# **One of the striking aspects of this worrisome phenomenon is the high rate of success achieved by terrorist groups and individuals. Alarmingly, a staggering 89% of these attacks have been successful, resulting in a range of devastating consequences for the affected populations. This success rate underscores the effectiveness and persistence of these malicious actors in carrying out their destructive agendas.**
# 
# **Furthermore, the data reveals that the use of bombings and explosions as tactics in these attacks has inflicted the most casualties. These incidents not only lead to loss of life but also cause severe injuries and widespread damage to property and infrastructure. The prevalence of such tactics highlights the devastating impact of explosive devices and the need for comprehensive efforts to counteract the proliferation and use of explosives on a global scale.**
# 
# **As terrorism continues to pose a significant global challenge, addressing the root causes, enhancing intelligence and security measures, and promoting international cooperation remain crucial in mitigating the impact and working toward a more secure and peaceful world.**
