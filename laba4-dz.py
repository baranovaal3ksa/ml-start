import pandas as pd
df = pd.read_csv('titanic.csv')
print(df.shape); print(df.head()); df.info(); print(df.describe())

print('------------------------')

kids = df[df['age'] < 12]
print(len(kids), kids['survived'].mean())

print('------------------------')

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])
print(df.isna().sum())

print('------------------------')

print(df.groupby('sex')['survived'].mean())
print(df.groupby('pclass')['survived'].mean())
print(df.groupby(['pclass', 'sex'])['age'].mean().round(1))

print('------------------------')

import matplotlib.pyplot as plt, seaborn as sns
fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
ax[0].hist(df['age'], bins=30); ax[0].set_title('Возраст')
sns.boxplot(x='pclass', y='fare', data=df, ax=ax[1]); ax[1].set_title('Цена по классу')
sns.barplot(x='sex', y='survived', data=df, ax=ax[2]); ax[2].set_title('Выживаемость')

ax[3].hist(df[df['survived'] == 0]['fare'], bins=30, alpha=0.5, label='Не выжили', color='red')
ax[3].hist(df[df['survived'] == 1]['fare'], bins=30, alpha=0.5, label='Выжили', color='green')
ax[3].set_title('Выжившие и невыжившие'); ax[3].legend()

fig.tight_layout(); fig.savefig('titanic.png', dpi=120)
plt.show()

print('------------------------')

df['family'] = df['sibsp'] + df['parch']
df['sex_num'] = df['sex'].map({'female': 1, 'male': 0})
df.to_csv('data/clean.csv', index=False)

print('------------------------')

