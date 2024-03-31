import pandas as pd
import os
import json
import csv
import re
import numpy as np
from matplotlib.ticker import FuncFormatter
import altair as alt
import streamlit as st
from collections import defaultdict, Counter


# Function to load data from CSV Files
def load_data(game, nrows=None):
    file_path = os.path.join('data', f'{game}.csv')
    return pd.read_csv(file_path, nrows=nrows)

def choices_to_df(choices, hue):
    df = pd.DataFrame(choices, columns=['choices'])
    df['hue'] = hue
    df['hue'] = df['hue'].astype(str)
    return df

def process_dictator_data_and_create_visualization(df):
    # Get list of available games
    game_files = [file.split('.')[0] for file in os.listdir('data') if file.endswith('.csv')]
    selected_game = 'dictator'  # Example, you can change this to any game file you have

    # Load data for the selected game (5000 lines only)
    df = load_data(selected_game, nrows=5000)

    binrange = (0, 100)
    moves = []
    for _, record in df.iterrows():
        if record['Role'] != 'first': continue
        if int(record['Round']) > 1: continue
        if int(record['Total']) != 100: continue
        if record['move'] == 'None': continue
        if record['gameType'] != 'dictator': continue

        move = float(record['move'])
        if move < binrange[0] or move > binrange[1]: continue

        moves.append(move)

    df_dictator_human = choices_to_df(moves, 'Human')

    # Example data for ChatGPT-4
    choices_gpt4 = [60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70]
    df_dictator_gpt4 = choices_to_df(choices_gpt4, hue=str('ChatGPT-4'))

    # Example data for ChatGPT-3
    choices_gpt3 = [25, 35, 70, 30, 20, 25, 40, 80, 30, 30, 40, 30, 30, 30, 30, 30, 40, 40, 30, 30, 40, 30, 60, 20, 40, 25, 30, 30, 30]
    df_dictator_turbo = choices_to_df(choices_gpt3, hue=str('ChatGPT-3'))

    # Concatenate all three dataframes
    alt_df = pd.concat([df_dictator_human, df_dictator_gpt4, df_dictator_turbo])

    # Altair plot for visualizing the distributions with interactivity
    alt_chart = alt.Chart(alt_df).mark_area(opacity=0.3).encode(
        x=alt.X('choices:Q', bin=alt.Bin(maxbins=20), title='Split offered ($)'),
        y=alt.Y('count():Q', stack=None, title='Density'),
        color=alt.Color('hue:N', title='Model')
    ).properties(
        width=600,
        height=200
    ).resolve_scale(
        y='independent'
    ).interactive()  # Add interactivity for zooming and panning

    # Display Altair chart
    return alt_chart

def process_ultimatum_strategy_data_and_create_visualization(df):
    selected_game = 'ultimatum_strategy'  # Example, you can change this to any game file you have

    # Load data for the selected game (5000 lines only)
    df = load_data(selected_game, nrows=5000)

    df = df[df['Role'] == 'player']
    df = df[df['Round'] == 1]
    df = df[df['Total'] == 100]
    df = df[df['move'] != 'None']
    df['propose'] = df['move'].apply(lambda x: eval(x)[0])
    df['accept'] = df['move'].apply(lambda x: eval(x)[1])
    df = df[(df['propose'] >= 0) & (df['propose'] <= 100)]
    df = df[(df['accept'] >= 0) & (df['accept'] <= 100)]

    df_ultimatum_1_human = choices_to_df(list(df['propose']), 'Human')
    df_ultimatum_2_human = choices_to_df(list(df['accept']), 'Human')


    # Model Data
    choices = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    df_ultimatum_1_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

    choices = [40, 40, 40, 30, 70, 70, 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 30, 30, 35, 50, 40, 70, 40, 60, 60, 70, 40, 50]
    df_ultimatum_1_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

    choices = [50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 50.0, 25.0, 50.0, 1.0, 1.0, 20.0, 50.0, 50.0, 50.0, 20.0, 50.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 20.0, 1.0] + [0, 1]
    df_ultimatum_2_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

    choices = [None, 50, 50, 50, 50, 30, None, None, 30, 33.33, 40, None, 50, 40, None, 1, 30, None, 10, 50, 30, 10, 30, None, 30, None, 10, 30, 30, 30]
    df_ultimatum_2_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))


    # Concatenate all three dataframes
    alt_df = pd.concat([df_ultimatum_1_human,
        df_ultimatum_1_gpt4,
        df_ultimatum_1_turbo])

    # Altair plot for visualizing the distributions with interactivity
    alt_chart = alt.Chart(alt_df).mark_area(opacity=0.3).encode(
        x=alt.X('choices:Q', bin=alt.Bin(maxbins=20), title='Proposal to give ($)'),
        y=alt.Y('count():Q', stack=None, title='Density'),
        color=alt.Color('hue:N', title='Model')
    ).properties(
        width=600,
        height=200
    ).resolve_scale(
        y='independent'
    ).interactive()  # Add interactivity for zooming and panning

    # Display Altair chart
    return alt_chart

def process_ultimatum_strategy_accept_data_and_create_visualization(df):
    selected_game = 'ultimatum_strategy'  # Example, you can change this to any game file you have

    # Load data for the selected game (5000 lines only)
    df = load_data(selected_game, nrows=5000)

    df = df[df['Role'] == 'player']
    df = df[df['Round'] == 1]
    df = df[df['Total'] == 100]
    df = df[df['move'] != 'None']
    df['propose'] = df['move'].apply(lambda x: eval(x)[0])
    df['accept'] = df['move'].apply(lambda x: eval(x)[1])
    df = df[(df['propose'] >= 0) & (df['propose'] <= 100)]
    df = df[(df['accept'] >= 0) & (df['accept'] <= 100)]

    df_ultimatum_1_human = choices_to_df(list(df['propose']), 'Human')
    df_ultimatum_2_human = choices_to_df(list(df['accept']), 'Human')


    # Model Data
    choices = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    df_ultimatum_1_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

    choices = [40, 40, 40, 30, 70, 70, 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 30, 30, 35, 50, 40, 70, 40, 60, 60, 70, 40, 50]
    df_ultimatum_1_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

    choices = [50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 50.0, 25.0, 50.0, 1.0, 1.0, 20.0, 50.0, 50.0, 50.0, 20.0, 50.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 20.0, 1.0] + [0, 1]
    df_ultimatum_2_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

    choices = [None, 50, 50, 50, 50, 30, None, None, 30, 33.33, 40, None, 50, 40, None, 1, 30, None, 10, 50, 30, 10, 30, None, 30, None, 10, 30, 30, 30]
    df_ultimatum_2_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))
    alt_df = pd.concat([df_ultimatum_2_human,
        df_ultimatum_2_gpt4,
        df_ultimatum_2_turbo])

    # Altair plot for visualizing the distributions with interactivity
    alt_chart = alt.Chart(alt_df).mark_area(opacity=0.3).encode(
        x=alt.X('choices:Q', bin=alt.Bin(maxbins=20), title='Minimum propsal to accept ($)'),
        y=alt.Y('count():Q', stack=None, title='Density'),
        color=alt.Color('hue:N', title='Model')
    ).properties(
        width=600,
        height=200
    ).resolve_scale(
        y='independent'
    ).interactive()  # Add interactivity for zooming and panning

    # Display Altair chart
    return alt_chart

def process_trust_investment_data_and_create_visualization(df):
    game_files = [file.split('.')[0] for file in os.listdir('data') if file.endswith('.csv')]
    selected_game = 'trust_investment'  # Example, you can change this to any game file you have
    binrange = (0, 100)
    moves_1 = []
    moves_2 = defaultdict(list)

    # Load data for the selected game (5000 lines only)
    df = load_data(selected_game, nrows=5000)

    for _, record in df.iterrows():
        if int(record['Round']) > 1: continue

        if record['Role'] != 'first': continue
        if int(record['Round']) > 1: continue
        if record['move'] == 'None': continue
        if record['gameType'] != 'trust_investment': continue

        if record['Role'] == 'first':
            move = float(record['move'])
            if move < binrange[0] or \
                move > binrange[1]: continue
            moves_1.append(move)
        elif record['Role'] == 'second':
            inv, ret = eval(record['roundResult'])
            if ret < 0 or \
                ret > inv * 3: continue
            moves_2[inv].append(ret)
        else: continue

    df_trust_1_human = choices_to_df(moves_1, 'Human')
    df_trust_2_human = choices_to_df(moves_2[10], 'Human')
    df_trust_3_human = choices_to_df(moves_2[50], 'Human')
    df_trust_4_human = choices_to_df(moves_2[100], 'Human')



    # Model Data
    choices = [50.0, 50.0, 40.0, 30.0, 50.0, 50.0, 40.0, 50.0, 50.0, 50.0, 50.0, 50.0, 30.0, 30.0, 50.0, 50.0, 50.0, 40.0, 40.0, 50.0, 50.0, 50.0, 50.0, 40.0, 50.0, 50.0, 50.0, 50.0] 
    df_trust_1_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

    choices = [50.0, 50.0, 30.0, 30.0, 30.0, 60.0, 50.0, 40.0, 20.0, 20.0, 50.0, 40.0, 30.0, 20.0, 30.0, 20.0, 30.0, 60.0, 50.0, 30.0, 50.0, 20.0, 20.0, 30.0, 50.0, 30.0, 30.0, 50.0, 40.0] + [30]
    df_trust_1_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

    choices = [20.0, 20.0, 20.0, 20.0, 15.0, 15.0, 15.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 15.0, 20.0, 15.0, 15.0, 15.0, 15.0, 15.0, 20.0, 20.0, 15.0]
    df_trust_2_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

    choices = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 25.0, 30.0, 30.0, 20.0, 25.0, 30.0, 20.0, 20.0, 18.0] + [20, 20, 20, 25, 25, 25, 30]
    df_trust_2_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

    choices = [100.0, 75.0, 75.0, 75.0, 75.0, 75.0, 100.0, 75.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 75.0, 100.0, 75.0, 75.0, 75.0, 100.0, 100.0, 100.0, 75.0, 100.0, 100.0, 100.0, 100.0, 75.0, 100.0, 75.0]
    df_trust_3_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

    choices = [150.0, 100.0, 150.0, 150.0, 50.0, 150.0, 100.0, 150.0, 100.0, 100.0, 100.0, 150.0] + [100, 100, 100, 100, 100, 100, 100, 100]
    df_trust_3_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

    choices = [200.0, 200.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 200.0, 200.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]
    df_trust_4_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))

    choices = [225.0, 225.0, 300.0, 300.0, 220.0, 300.0, 250.0] + [200, 200, 250, 200, 200]
    df_trust_4_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))

    # Concatenate all three dataframes
    alt_df = pd.concat([df_trust_1_human,
        df_trust_1_gpt4,
        df_trust_1_turbo])

    # Altair plot for visualizing the distributions with interactivity
    alt_chart = alt.Chart(alt_df).mark_area(opacity=0.3).encode(
        x=alt.X('choices:Q', bin=alt.Bin(maxbins=20), title='Investment ($)'),
        y=alt.Y('count():Q', stack=None, title='Density'),
        color=alt.Color('hue:N', title='Model')
    ).properties(
        width=600,
        height=200
    ).resolve_scale(
        y='independent'
    ).interactive()  # Add interactivity for zooming and panning

# Display Altair chart
    return alt_chart

def process_public_goods_data_and_create_visualization(df):
    game_files = [file.split('.')[0] for file in os.listdir('data') if file.endswith('.csv')]
    selected_game = 'public_goods_linear_water'  # Example, you can change this to any game file you have


    # Load data for the selected game (5000 lines only)
    df = load_data(selected_game, nrows=5000)

    df = df[df['Role'] == 'contributor']
    df = df[df['Round'] <= 3]
    df = df[df['Total'] == 20]
    df = df[df['groupSize'] == 4]
    df = df[df['move'] != None]
    df = df[(df['move'] >= 0) & (df['move'] <= 20)]
    df = df[df['gameType'] == 'public_goods_linear_water']

    round_1 = df[df['Round'] == 1]['move']
    round_2 = df[df['Round'] == 2]['move']
    round_3 = df[df['Round'] == 3]['move']
    print(len(round_1), len(round_2), len(round_3))
    df_PG_human = pd.DataFrame({
    'choices': list(round_1)
    })
    df_PG_human['hue'] = 'Human'
    df_PG_human


    file_names = [
    # 'records/PG_basic_turbo_2023_05_09-02_49_09_AM.json',
    # 'records/PG_basic_turbo_loss_2023_05_09-03_59_49_AM.json'
    'records/PG_basic_gpt4_2023_05_09-11_15_42_PM.json',
    'records/PG_basic_gpt4_loss_2023_05_09-10_44_38_PM.json',
    ]

    choices = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            choices += json.load(f)['choices']
            choices_baseline = choices

    choices = [tuple(x)[0] for x in choices]
    df_PG_turbo = choices_to_df(choices, hue=str('ChatGPT-3'))
    # df_PG_turbo.head()
    df_PG_gpt4 = choices_to_df(choices, hue=str('ChatGPT-4'))
    df_PG_gpt4.head()


    # Concatenate all three dataframes
    alt_df = pd.concat([df_PG_human,
        df_PG_gpt4,
        df_PG_turbo])

    # Altair plot for visualizing the distributions with interactivity
    alt_chart = alt.Chart(alt_df).mark_area(opacity=0.3).encode(
        x=alt.X('choices:Q', bin=alt.Bin(maxbins=20), title='Contribution ($)'),
        y=alt.Y('count():Q', stack=None, title='Density'),
        color=alt.Color('hue:N', title='Model')
    ).properties(
        width=600,
        height=200
    ).resolve_scale(
        y='independent'
    ).interactive()  # Add interactivity for zooming and panning

    # Display Altair chart
    return alt_chart

def process_bomb_risk_data_and_create_visualization(df):
    # Human Data

    game_files = [file.split('.')[0] for file in os.listdir('data') if file.endswith('.csv')]
    selected_game = 'bomb_risk'  # Example, you can change this to any game file you have


    # Load data for the selected game (5000 lines only)
    df = load_data(selected_game, nrows=5000)

    df = df[df['Role'] == 'player']
    df = df[df['gameType'] == 'bomb_risk']
    df.sort_values(by=['UserID', 'Round'])

    prefix_to_choices_human = defaultdict(list)
    prefix_to_IPW = defaultdict(list)
    prev_user = None
    prev_move = None
    prefix = ''
    bad_user = False
    for _, row in df.iterrows():
        if bad_user: continue
        if row['UserID'] != prev_user:
            prev_user = row['UserID']
            prefix = ''
            bad_user = False

        move = row['move']
        if move < 0 or move > 100:
            bad_users = True
            continue
        prefix_to_choices_human[prefix].append(move)

        if len(prefix) == 0:
            prefix_to_IPW[prefix].append(1)
        elif prefix[-1] == '1':
            prev_move = min(prev_move, 98)
            prefix_to_IPW[prefix].append(1./(100 - prev_move))
        elif prefix[-1] == '0':
            prev_move = max(prev_move, 1)
            prefix_to_IPW[prefix].append(1./(prev_move))
        else: assert False
    
        prev_move = move

        prefix += '1' if row['roundResult'] == 'SAFE' else '0'


    # Model Data
    prefix_to_choices_model = defaultdict(lambda : defaultdict(list))
    for model in ['ChatGPT-4', 'ChatGPT-3']:
        if model == 'ChatGPT-4':
            file_names = [
            'bomb_gpt4_2023_05_15-12_13_51_AM.json'
            ]
        elif model == 'ChatGPT-3':
            file_names = [
            'bomb_turbo_2023_05_14-10_45_50_PM.json'
            ]

        choices = []
        scenarios = []
        for file_name in file_names:
            with open(os.path.join('records', file_name), 'r') as f:
                records = json.load(f)
                choices += records['choices']
                scenarios += records['scenarios']

        assert len(scenarios) == len(choices)
        print('loaded %i valid records' % len(scenarios))

        prefix_to_choice = defaultdict(list)
        prefix_to_result = defaultdict(list)
        prefix_to_pattern = defaultdict(Counter)
        wrong_sum = 0
        for scenarios_tmp, choices_tmp in zip(scenarios, choices):

            result = 0
            for i, scenario in enumerate(scenarios_tmp):
                prefix = tuple(scenarios_tmp[:i])
                prefix = ''.join([str(x) for x in prefix])
                choice = choices_tmp[i]
            
                prefix_to_choice[prefix].append(choice)
                prefix_to_pattern[prefix][tuple(choices_tmp[:-1])] += 1

                prefix = tuple(scenarios_tmp[:i+1])
                if scenario == 1:
                    result += choice
                prefix_to_result[prefix].append(result)

        print('# of wrong sum:', wrong_sum)
        print('# of correct sum:', len(scenarios) - wrong_sum)

        prefix_to_choices_model[model] = prefix_to_choice

    prefix = ''
    df_bomb_human = choices_to_df(prefix_to_choices_human[prefix], hue='Human')
    df_bomb_human['weight'] = prefix_to_IPW[prefix]
    df_bomb_models = pd.concat([choices_to_df(
            prefix_to_choices_model[model][prefix], hue=model
        ) for model in prefix_to_choices_model]
    )
    df_bomb_models['weight'] = 1


    # Concatenate all three dataframes
    alt_df = pd.concat([df_bomb_human,
        df_bomb_models])

    # Altair plot for visualizing the distributions with interactivity
    alt_chart = alt.Chart(alt_df).mark_area(opacity=0.3).encode(
        x=alt.X('choices:Q', bin=alt.Bin(maxbins=20), title='# of boxes opened'),
        y=alt.Y('count():Q', stack=None, title='Density'),
        color=alt.Color('hue:N', title='Model')
    ).properties(
        width=600,
        height=200
    ).resolve_scale(
        y='independent'
    ).interactive()  # Add interactivity for zooming and panning

    # Display Altair chart
    return alt_chart
# Function to create Altair chart
# def create_altair_chart(df, x_title, y_title):
#     alt_chart = alt.Chart(df).mark_area(opacity=0.3).encode(
#         x=alt.X('choices:Q', bin=alt.Bin(maxbins=20), title=x_title),
#         y=alt.Y('count():Q', stack=None, title=y_title),
#         color=alt.Color('hue:N', title='Model')
#     ).properties(
#         width=600,
#         height=200
#     ).resolve_scale(
#         y='independent'
#     ).interactive()  # Add interactivity for zooming and panning
#     return alt_chart

# # Function to create histogram from choices data
# def choices_to_df(choices, hue):
#     return pd.DataFrame({
#         'choices': choices,
#         'hue': hue
#     })

# # Load data for the selected game (5000 lines only)
# # Load data for the selected game (5000 lines only)
# def load_game_data(selected_game):
#     if selected_game == 'dictator':
#         # Load dictator game data
#         df = load_data(selected_game, nrows=5000)
#         # Further data processing to get required columns
#         binrange = (0, 100)
#         moves = []
#         for _, record in df.iterrows():
#             if record['Role'] != 'first': continue
#             if int(record['Round']) > 1: continue
#             if int(record['Total']) != 100: continue
#             if record['move'] == 'None': continue
#             if record['gameType'] != 'dictator': continue

#             move = float(record['move'])
#             if move < binrange[0] or move > binrange[1]: continue

#             moves.append(move)

#         df_dictator_human = choices_to_df(moves, 'Human')

#         # Example data for ChatGPT-4
#         choices_gpt4 = [60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70, 50, 80, 60, 70]
#         df_dictator_gpt4 = choices_to_df(choices_gpt4, hue=str('ChatGPT-4'))

#         # Example data for ChatGPT-3
#         choices_gpt3 = [25, 35, 70, 30, 20, 25, 40, 80, 30, 30, 40, 30, 30, 30, 30, 30, 40, 40, 30, 30, 40, 30, 60, 20, 40, 25, 30, 30, 30]
#         df_dictator_turbo = choices_to_df(choices_gpt3, hue=str('ChatGPT-3'))

#         # Concatenate all three dataframes
#         alt_df = pd.concat([df_dictator_human, df_dictator_gpt4, df_dictator_turbo])

#         alt_chart = create_altair_chart(alt_df, 'Split offered ($)', 'Density')

#     elif selected_game == 'ultimatum_strategy':
#         # Load ultimatum strategy game data
#         df = load_data(selected_game, nrows=5000)
#         # Further data processing to get required columns
#         df = df[df['Role'] == 'player']
#         df = df[df['Round'] == 1]
#         df = df[df['Total'] == 100]
#         df = df[df['move'] != 'None']
#         df['propose'] = df['move'].apply(lambda x: eval(x)[0])
#         df['accept'] = df['move'].apply(lambda x: eval(x)[1])
#         df = df[(df['propose'] >= 0) & (df['propose'] <= 100)]
#         df = df[(df['accept'] >= 0) & (df['accept'] <= 100)]

#         df_ultimatum_1_human = choices_to_df(list(df['propose']), 'Human')
#         df_ultimatum_2_human = choices_to_df(list(df['accept']), 'Human')

#         # Model Data for ChatGPT-4
#         choices_gpt4 = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
#         df_ultimatum_1_gpt4 = choices_to_df(choices_gpt4, hue=str('ChatGPT-4'))

#         choices_gpt4 = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]

#                 # Model Data for ChatGPT-4 (continued)
#         df_ultimatum_2_gpt4 = choices_to_df(choices_gpt4, hue=str('ChatGPT-4'))

#         # Model Data for ChatGPT-3
#         choices_gpt3 = [40, 40, 40, 30, 70, 70, 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 30, 30, 35, 50, 40, 70, 40, 60, 60, 70, 40, 50]
#         df_ultimatum_1_turbo = choices_to_df(choices_gpt3, hue=str('ChatGPT-3'))

#         choices_gpt3 = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
#         df_ultimatum_2_turbo = choices_to_df(choices_gpt3, hue=str('ChatGPT-3'))

#         # Concatenate all three dataframes
#         alt_df_ultimatum_1 = pd.concat([df_ultimatum_1_human, df_ultimatum_1_gpt4, df_ultimatum_1_turbo])
#         alt_df_ultimatum_2 = pd.concat([df_ultimatum_2_human, df_ultimatum_2_gpt4, df_ultimatum_2_turbo])

#         # Altair plot for visualizing the distributions with interactivity
#         alt_chart_ultimatum_1 = create_altair_chart(alt_df_ultimatum_1, 'Proposal to give ($)', 'Density')
#         alt_chart_ultimatum_2 = create_altair_chart(alt_df_ultimatum_2, 'Minimum proposal to accept ($)', 'Density')

#         return alt_chart_ultimatum_1, alt_chart_ultimatum_2

#     elif selected_game == 'trust_investment':
#         # Load trust investment game data
#         df = load_data(selected_game, nrows=5000)
#         # Further data processing to get required columns
#         binrange = (0, 100)
#         moves_1 = []
#         moves_2 = defaultdict(list)

#         for _, record in df.iterrows():
#             if int(record['Round']) > 1: continue

#             if record['Role'] != 'first': continue
#             if int(record['Round']) > 1: continue
#             if record['move'] == 'None': continue
#             if record['gameType'] != 'trust_investment': continue

#             if record['Role'] == 'first':
#                 move = float(record['move'])
#                 if move < binrange[0] or move > binrange[1]: continue
#                 moves_1.append(move)
#             elif record['Role'] == 'second':
#                 inv, ret = eval(record['roundResult'])
#                 if ret < 0 or ret > inv * 3: continue
#                 moves_2[inv].append(ret)
#             else: continue

#         df_trust_1_human = choices_to_df(moves_1, 'Human')
#         df_trust_2_human = choices_to_df(moves_2[10], 'Human')
#         df_trust_3_human = choices_to_df(moves_2[50], 'Human')
#         df_trust_4_human = choices_to_df(moves_2[100], 'Human')

#         # Model Data for ChatGPT-4
#         choices_gpt4_1 = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
#         df_trust_1_gpt4 = choices_to_df(choices_gpt4_1, hue=str('ChatGPT-4'))

#         choices_gpt4_2 = [50.0, 50.0, 30.0, 30.0, 30.0, 60.0, 50.0, 40.0, 20.0, 20.0, 50.0, 40.0, 30.0, 20.0, 30.0, 20.0, 30.0, 60.0, 50.0, 30.0, 50.0, 20.0, 20.0, 30.0, 50.0, 30.0, 30.0, 50.0, 40.0] + [30]
#         df_trust_1_gpt4 = choices_to_df(choices_gpt4_2, hue=str('ChatGPT-4'))

#         # Concatenate all three dataframes
#         alt_df_trust_1 = pd.concat([df_trust_1_human, df_trust_1_gpt4])
#         alt_df_trust_2 = pd.concat([df_trust_2_human, df_trust_2_gpt4])
#         alt_df_trust_3 = pd.concat([df_trust_3_human, df_trust_3_gpt4])
#         alt_df_trust_4 = pd.concat([df_trust_4_human, df_trust_4_gpt4])

#         # Altair plot for visualizing the distributions with interactivity
#         alt_chart_trust_1 = create_altair_chart(alt_df_trust_1, 'Investment ($)', 'Density')
#         alt_chart_trust_2 = create_altair_chart(alt_df_trust_2, 'Return to investor ($)', 'Density')
#         alt_chart_trust_3 = create_altair_chart(alt_df_trust_3, 'Investment ($)', 'Density')
#         alt_chart_trust_4 = create_altair_chart(alt_df_trust_4, 'Return to investor ($)', 'Density')

#         return alt_chart_trust_1, alt_chart_trust_2, alt_chart_trust_3, alt_chart_trust_4


# Define Streamlit app layout
def main():
    st.title('How does AI chatbot behavior compare to humans?')

    # Add description or instructions if needed
    st.write("Distributions of choices of ChatGPT-4, ChatGPT-3, and human subjects in each game: (A) Dictator; (B) Ultimatum as proposer; (C) Ultimatum as responder; (D) Trust as investor; (E) Trust as banker; (F) Public Goods; (G) Bomb Risk; (H) Prisoner’s Dilemma. Both chatbots’ distributions are more tightly clustered and contained within the range of the human distribution. ChatGPT-4 makes more concentrated decisions than ChatGPT-3. Compared to the human distribution, on average, the AIs make a more generous split to the other player as a dictator, as the proposer in the Ultimatum Game, and as the Banker in the Trust Game, on average. ChatGPT-4 proposes a strictly equal split of the endowment both as a dictator or as the proposer in the Ultimatum Game. Both AIs make a larger investment in the Trust Game and a larger contribution to the Public Goods project, on average. They are more likely to cooperate with the other player in the first round of the Prisoner’s Dilemma Game. Both AIs predominantly make a payoff-maximization decision in a single-round Bomb Risk Game. Density is the normalized count such that the total area of the histogram equals 1.")

    # Get list of available games
    game_files = [file.split('.')[0] for file in os.listdir('data') if file.endswith('.csv')]

    # Add a dropdown for selecting the game
    selected_game = st.selectbox("Select a game", game_files)

    # Load data for the selected game
    df = load_data(selected_game, nrows=5000)


    # Process data and create visualizations
    if selected_game == 'dictator':
        alt_chart = process_dictator_data_and_create_visualization(df)
    elif selected_game == 'ultimatum_strategy':
        alt_chart = process_ultimatum_strategy_data_and_create_visualization(df)
    elif selected_game == 'trust_investment':
        alt_chart = process_trust_investment_data_and_create_visualization(df)
    elif selected_game == 'public_goods_linear_water':
        alt_chart = process_public_goods_data_and_create_visualization(df)
    elif selected_game == 'bomb_risk':
        alt_chart = process_bomb_risk_data_and_create_visualization(df)
    else:
        alt_chart = None

    # Display the visualization
    if alt_chart is not None:
        st.write("Visualization")
        st.altair_chart(alt_chart, use_container_width=True)
    else:
        st.write("No visualization available for the selected game.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
