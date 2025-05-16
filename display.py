import pandas as pd
import streamlit as st
import altair as alt
from seaborn import color_palette
from sys import argv

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

if __name__ == '__main__':
  if len(argv) > 2:
    st.write("Usage: python display.py [filename]")
    st.write("If no filename is provided, the default file 'provisional.csv' will be used.")
  filename = argv[1] if len(argv) > 1 else 'provisional.csv'

  df = pd.read_csv(filename)
  colors_cluster = color_generator(df['cluster'].unique(), palette='viridis')
  df['datetime'] = pd.to_datetime(df['datetime'])

  st.title('ARCHER - Machine Translation Evaluation Results')
  st.logo('images/archer.png', icon_image='images/archer-short.png', link='http://archer.prhlt.upv.es/')

  #################
  # LEADER BOARD
  #################
  with st.sidebar:
    st.header('Leader board')
    leaderborad = df[['position','name', 'cluster']].copy()
    st.table(leaderborad.sort_values('position').set_index('position'))

  ##################
  # FILTERS
  ##################
  # Select main metric
  metric = st.selectbox(
      'Select the metric to display',
      ('BLEU', 'TER')
  ).lower()

  # Range of clusters
  # num_clusters = max(df['cluster'])
  # best_cluster, worse_cluster = st.slider("Displayed clusters", 1, num_clusters, (1, max(df['cluster'])))
  # df = df[(df['cluster'] >= best_cluster) & (df['cluster'] <= worse_cluster)]

  #################################
  # CHARTS
  #################################
  bar_tab, scatter_tab, cluster_tab = st.tabs(['Leader board', 'Quality vs Time', 'Clusters average'])

  with bar_tab:
    bar = (
        alt.Chart(df)
          .mark_bar()
          .encode(
            x=alt.X('name:N', sort=None, title='Participant'),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score'),
            color=colors_cluster,
            tooltip=['name','cluster',f'{metric}']
          )
          .properties(title=f'{metric.upper()} by Participant', width=700, height=400)
    )

    # Rotate the x-axis labels so they fit
    bar = bar.configure_axisX(labelAngle=-45)
    st.altair_chart(bar, use_container_width=True)

  with scatter_tab:
    chart = (
        alt.Chart(df)
          .mark_circle(size=80)
          .encode(
            x=alt.X('time:Q', title='Run time (s)'),
            y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score'),
            color=colors_cluster,
            tooltip=['name','cluster',f'{metric}','time']
          )
          .properties(title=f"{metric.upper()} vs Computation Time", width=700, height=400)
    )
    st.altair_chart(chart, use_container_width=True)

  with cluster_tab:
    cluster_df = df.groupby('cluster').agg(
        {
            'bleu': 'mean',
            'ter': 'mean'}).reset_index()
    bar = (
      alt.Chart(cluster_df)
      .mark_bar()
      .encode(
          x=alt.X('cluster:N', sort=None, title='Cluster'),
          y=alt.Y(f'{metric}:Q', title=f'{metric.upper()} Score'),
          color=colors_cluster,
          tooltip=['cluster',f'{metric}']
      )
      .properties(title=f'Average {metric.upper()} by Cluster', width=700, height=400)
    )
    bar = bar.configure_axisX(labelAngle=0)
    st.altair_chart(bar, use_container_width=True)

  #################################
  # TABLES
  #################################
  st.write(df)
  st.download_button(
      label="Download data",
      data=df.to_csv(index=False).encode('utf-8'),
      file_name='archer_results.csv',
      mime='text/csv',
  )