## Cell 1 (code) [no output] [2]

```python
import pandas as pd
import regex as re
import os
from pathlib import Path
```

## Cell 2 (code) [3]

```python
df = pd.read_csv('./DATASETS_excalidraw/file_lifecycle_events.csv', encoding='utf-8')
df.columns
```

**Result:**
```text
Index(['filepath', 'event_type', 'event_datetime', 'datetime_last_modified',
       'commit_hash', 'author_name', 'author_email', 'commit_subject',
       'additions', 'deletions', 'is_binary', 'old_path', 'new_path',
       'source_path'],
      dtype='object')
```

## Cell 3 (code) [12]

```python
df.sample()
```

**Result:**
```text
filepath event_type  \
16779  packages/{excalidraw/element => element/src}/t...   modified   

            event_datetime datetime_last_modified  \
16779  2025-03-26T17:24:59    2025-04-14T09:18:11   

                                    commit_hash  author_name  \
16779  432a46ef9ee13afe47530e857a259acde4d0fd59  Marcel Mraz   

                author_email  \
16779  marcel@excalidraw.com   

                                          commit_subject  additions  \
16779  refactor: separate elements logic into a stand...          5   

       deletions  is_binary  old_path  new_path  source_path  
16779          4      False       NaN       NaN          NaN
```

## Cell 4 (code) [ ]

```python
df[df['filepath'].apply(lambda x: '=>' in x)]['filepath'].sample(10) # what are those arrows `=>` ??
```

**Result:**
```text
41831    {src => packages/excalidraw}/components/Export...
41354    {public/fonts => packages/excalidraw/fonts/ass...
41563    {src => packages/excalidraw}/actions/shortcuts.ts
12445    packages/excalidraw/fonts/{woff2 => }/Xiaolai/...
42436    {src => packages/excalidraw}/components/dropdo...
43708    {src/element => packages/element/src}/transfor...
41632      {src => packages/excalidraw}/components/App.tsx
42941      {src => packages/excalidraw}/locales/bg-BG.json
42563          {src => packages/excalidraw}/css/theme.scss
13711    packages/excalidraw/renderer/{renderScene.ts =...
Name: filepath, dtype: object
```

## Cell 5 (code) [no output] [ ]

```python

```

## Cell 6 (code) [no output] [ ]

```python

```

