import pandas as pd
import streamlit as st
import altair as alt
from seaborn import color_palette
from sys import argv

def get_metrics(list_columns):
    """
    Get the list of metrics from the dataframe columns.
    """
    non_metric = ['name','time', 'datetime', 'comment', 'task', 'subtask', 'position']
    return [col for col in list_columns if col not in non_metric and not col.startswith('cluster_')]

def color_generator(categories, label='cluster', palette='crest'):
    """
    Generate a list of n colors.
    """
    palette = color_palette(palette, len(categories)).as_hex()
    return alt.Color(
            f'{label}:N',
            scale=alt.Scale(
              domain=categories,
              range=palette
            ),
            legend=alt.Legend(title="Cluster")
         )

def bar_leaderboard(df, metric, colors_cluster):
    """
    Generate a bar chart for the leaderboard.
    """
    top = 1.1 * df[metric].max()
    bottom = 0.9 * df[metric].min()

    # Create a conditional rule definition
    rule = alt.Chart(df[df['comment'] == 'Baseline']).mark_rule(strokeDash=[8, 8], color='red', size=2).encode(
        x=alt.X('name:N', sort=None, title='Participant'),
    )

    bar = (
        alt.Chart(df)
        .mark_bar(clip=True)
        .encode(
            x=alt.X('name:N', sort=None, title='Participant'),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score', scale=alt.Scale(domain=[bottom, top])),
            color=colors_cluster,
            tooltip=['name', f'cluster_{metric}', f'{metric}'],
        )
        .properties(title=f'{metric.upper()} by Participant', width=700, height=400)
    )

    # Layer the bar chart and the rule
    chart = alt.layer(bar, rule).configure_axisX(labelAngle=-90)
    st.altair_chart(chart, use_container_width=True)
  
def scatter_chart(df,metric, colors_cluster):
    """
    Generate a scatter chart for the leaderboard.
    """
    top = 1.1 * df[metric].max()
    bottom = 0.9 * df[metric].min()
    right = 1.1 * df['time'].max()
    left = 0.9 * df['time'].min()
    chart = (
        alt.Chart(df)
          .mark_circle(size=80)
          .encode(
            x=alt.X('time:Q', title='Run time (s)', scale=alt.Scale(domain=[left, right])),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score', scale=alt.Scale(domain=[bottom, top])),
            color=colors_cluster,
            tooltip=['name',f'cluster_{metric}',f'{metric}','time']
          )
          .properties(title=f"{metric.upper()} vs Computation Time", width=700, height=400)
    )
    st.altair_chart(chart, use_container_width=True)

def cluster_chart(df, metric, colors_cluster, mode):
    """
    Generate a cluster chart for the leaderboard.
    """
    conf = {}
    conf[metric] = 'mean'
    conf['comment'] = lambda x: 'Baseline' if any(x == 'Baseline') else None
    cluster_df = df.groupby(f'cluster_{metric}').agg(conf).reset_index()
    top = 1.1 * cluster_df[metric].max()
    bottom = 0.9 * cluster_df[metric].min()
    bar = (
      alt.Chart(cluster_df)
      .mark_bar(clip=True)
      .encode(
          x=alt.X(f'cluster_{metric}:N', sort=None, title='Cluster'),
          y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score',scale=alt.Scale(domain=[bottom, top])),
          color=colors_cluster,
          tooltip=[f'cluster_{metric}',f'{metric}']
      )
      .properties(title=f'Average {metric.upper()} by Cluster', width=700, height=400)
    )

    rule = alt.Chart(cluster_df[cluster_df['comment'] == 'Baseline']).mark_rule(strokeDash=[8, 8], color='red', size=2).encode(
        x=alt.X(f'cluster_{metric}:N', sort=None, title='Cluster'),
    )
    chart = alt.layer(bar, rule).configure_axisX(labelAngle=0)
    st.altair_chart(chart, use_container_width=True)

def time_chart(df):
    """
    Generate a time pie chart for the leaderboard.
    """
    df = df.sort_values('datetime')
    legend = alt.Legend(title="Participant")
    chart = (
        alt.Chart(df)
          .mark_arc(innerRadius=50, outerRadius=100)
          .encode(
              theta=alt.Theta('time:Q', title='Run time (s)'),
              color=alt.Color('name:N', legend=legend),
              tooltip=['name','datetime','time']
          )
          .properties(title="Computation Time", width=700, height=400)
    )
    st.altair_chart(chart, use_container_width=True)

def general_view(df):
    """
    General view of the dataframe.
    """
    metrics = get_metrics(df.columns)
    scores = [{'metric': metric, 'scores': df[metric].values.tolist()} for metric in metrics]
    sc_df = pd.DataFrame(scores)
    st.dataframe(sc_df,
                 column_config={
                     "metric": st.column_config.TextColumn(
                         "Metric",
                         help="The evaluation metric",
                     ),
                     "scores": st.column_config.BarChartColumn(
                         "Scores",
                         help="The scores for each participant",
                     ),
                 },
                 hide_index=True,
                 use_container_width=True)

def main():
    if len(argv) != 3:
      st.write("Usage: python display.py <filename.csv> <mode>")
    filename = argv[1]
    MODE = argv[2]

    if MODE == 'mt':
      st.title('Machine Translation Evaluation Results')
    elif MODE == 'dr':
      st.title('Handwritten Text Recognition Evaluation Results')
    else:
      print("Invalid mode. Available modes: ['mt', 'dr']")
      exit(1)
    df = pd.read_csv(filename)

    ##################
    # FILTERS
    ##################
    # Select main metric
    #opt = ('BLEU', 'TER') if MODE == 'mt' else ('BWER', 'WER')
    opt = tuple(m.upper() for m in get_metrics(df.columns))
    metric = st.selectbox(
        'Select the metric to display',
        opt
    ).lower()

    with st.expander("Filter proposals", expanded=False):
        participants = df['name'].unique().tolist()
        selected_participants = st.multiselect(
            'Select participants to display',
            participants,
            default=participants
        )
        df = df[df['name'].isin(selected_participants)]

    if df.empty:
        st.write("No data to display. Please adjust the filters.")
        return

    colors_cluster = color_generator(df[f'cluster_{metric}'].unique(), label=f'cluster_{metric}', palette='viridis')
    df['datetime'] = pd.to_datetime(df['datetime'])

    #################
    # LEADER BOARD
    #################
    with st.sidebar:
      st.header('Leader board')
      leaderborad = df[['position','name', f'cluster_{metric}']].copy()
      st.table(leaderborad.sort_values('position').set_index('position'))

    #################################
    # CHARTS
    #################################
    bar_tab, scatter_tab, cluster_tab, time_tab = st.tabs(['Leader board', 'Quality vs Time', 'Clusters average', 'Time'])

    with bar_tab:
      bar_leaderboard(df, metric, colors_cluster)

    with scatter_tab:
      scatter_chart(df, metric, colors_cluster)

    with cluster_tab:
      cluster_chart(df, metric, colors_cluster, MODE)

    with time_tab:
      time_chart(df)

    #################################
    # TABLES
    #################################
    with st.expander("Show data", expanded=False):
      st.write(df)
      st.download_button(
          label="Download data",
          data=df.to_csv(index=False).encode('utf-8'),
          file_name='archer_results.csv',
          mime='text/csv',
      )

if __name__ == '__main__':
    main()