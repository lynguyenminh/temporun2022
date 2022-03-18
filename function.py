def remove_space(string):
    return " ".join(string.split())

def check_query_in_song(temp_query):
  for i in range(len(data)):
    if temp_query in data[i]['lyrics']:
      return data[i]['song_id']
  return 'No'
