# PyGuitarPro API Guide

## 핵심 구조

```
Song
├── title, artist, album, tempo
├── measureHeaders[] (마디 헤더)
└── tracks[]
    └── Track
        ├── name, channel, strings[], offset (capo!)
        └── measures[]
            └── Measure
                └── voices[]
                    └── Voice
                        └── beats[]
                            └── Beat
                                ├── duration, status
                                └── notes[]
                                    └── Note
                                        ├── string, value, velocity
                                        └── effect (hammer, slide, bend, etc.)
```

## 핵심 설정

### 1. 카포 (Capo)
```python
track.offset = 2  # 2프렛 카포
```
> 악보는 0프렛으로 표시되지만 실제 소리는 카포 적용된 음으로 재생

### 2. 튜닝 (Tuning)
```python
# Standard E: [64, 59, 55, 50, 45, 40] (1번줄 E4 ~ 6번줄 E2)
# DADGAD:    [62, 57, 55, 50, 45, 38]
# Drop D:    [64, 59, 55, 50, 45, 38]

track.strings = [
    gp.GuitarString(1, 64),  # 1번줄 (가는 줄) E4
    gp.GuitarString(2, 59),  # 2번줄 B3
    gp.GuitarString(3, 55),  # 3번줄 G3
    gp.GuitarString(4, 50),  # 4번줄 D3
    gp.GuitarString(5, 45),  # 5번줄 A2
    gp.GuitarString(6, 40),  # 6번줄 (굵은 줄) E2
]
```

### 3. Duration (음표 길이)
```python
# value: 1=온음표, 2=2분음표, 4=4분음표, 8=8분음표, 16=16분음표, 32=32분음표

beat.duration = gp.Duration(value=4)  # 4분음표
beat.duration.isDotted = True         # 점음표
beat.duration.tuplet.enters = 3       # 3잇단음표
beat.duration.tuplet.times = 2
```

## 이펙트 (Effects)

### Note Level Effects
```python
# Hammer-on / Pull-off
note.effect.hammer = True

# Slide
note.effect.slides = [gp.SlideType.shiftSlide]

# Bend
note.effect.bend = gp.BendEffect(
    type=gp.BendType.bend,
    value=100,
    points=[
        gp.BendPoint(0, 0),
        gp.BendPoint(6, 100),
        gp.BendPoint(12, 100)
    ]
)

# Vibrato
note.effect.vibrato = True

# Natural Harmonic
note.effect.harmonic = gp.NaturalHarmonic()

# Artificial Harmonic
note.effect.harmonic = gp.ArtificialHarmonic()

# Let Ring
note.effect.letRing = True

# Palm Mute
note.effect.palmMute = True

# Staccato
note.effect.staccato = True

# Ghost Note
note.type = gp.NoteType.ghost

# Dead Note (X)
note.type = gp.NoteType.dead

# Tie
note.type = gp.NoteType.tie
```

### Beat Level Effects
```python
# Slap (엄지 퍼커션)
beat.effect.slapEffect = gp.SlapEffect.slapping

# Pop
beat.effect.slapEffect = gp.SlapEffect.popping

# Stroke (스트럼 방향)
beat.stroke.direction = gp.BeatStrokeDirection.down
beat.stroke.value = gp.BeatStrokeDuration.sixteenth
```

### Grace Note (꾸밈음)
```python
grace = gp.GraceEffect()
grace.fret = 10
grace.duration = 2  # 1=1/16, 2=1/32, 3=1/64
grace.transition = gp.GraceEffectTransition.slide  # none, slide, hammer, bend
grace.isOnBeat = False  # False = 박자 앞에 붙음
note.effect.grace = grace
```

## AI 데이터 포맷 (권장 JSON 구조)

```json
{
  "metadata": {
    "title": "Song Title",
    "artist": "Artist Name",
    "tempo": 120
  },
  "instrument": {
    "tuning": [64, 59, 55, 50, 45, 40],
    "capo": 0
  },
  "measures": [
    {
      "beats": [
        {
          "duration": "quarter",
          "notes": [
            {
              "string": 1,
              "fret": 5,
              "technique": "hammer-on"
            }
          ]
        }
      ]
    }
  ]
}
```

## 코드 예시

```python
import guitarpro as gp

def create_song():
    song = gp.Song()
    song.title = "My Song"
    song.tempo = 120
    
    # Track
    track = gp.Track(song)
    track.name = "Guitar"
    track.offset = 2  # Capo 2
    track.strings = [gp.GuitarString(i+1, m) for i, m in enumerate([64,59,55,50,45,40])]
    
    # Measure Header
    header = gp.MeasureHeader()
    header.timeSignature = gp.TimeSignature(4, gp.Duration(4))
    song.measureHeaders.append(header)
    
    # Measure
    measure = gp.Measure(track, header)
    
    # Beat with notes
    beat = gp.Beat(measure.voices[0])
    beat.duration = gp.Duration(value=4)
    
    note = gp.Note(beat)
    note.string = 1
    note.value = 5
    note.velocity = 95
    note.effect.hammer = True
    beat.notes.append(note)
    
    measure.voices[0].beats.append(beat)
    track.measures.append(measure)
    song.tracks.append(track)
    
    gp.write(song, "output.gp5")
```

## 참고
- PyGuitarPro GitHub: https://github.com/Perlence/PyGuitarPro
- MIDI Note Numbers: https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
