# PyGuitarPro 참조 문서

**출처**: 
- GitHub: https://github.com/Perlence/PyGuitarPro
- 문서: https://pyguitarpro.readthedocs.io/en/stable/

---

## 📌 핵심 발견 (OmniTab 관련)

### 1. Song 생성 시 주의사항

```python
import guitarpro as gp

# Song() 생성 시 자동으로 생성되는 것들:
song = gp.Song()
# - song.tracks[0]: 기본 Track 1개
# - song.measureHeaders[0]: 기본 MeasureHeader 1개
# - track.measures[0]: 기본 Measure 1개

# ❌ 잘못된 방법: 새 Track을 append
track = gp.Track(song)
song.tracks.append(track)  # 2번째 트랙이 됨 → 저장 후 문제 발생

# ✅ 올바른 방법: 기존 Track 수정
track = song.tracks[0]
track.name = "My Guitar"
track.measures.clear()  # 기존 measure 삭제
```

---

## 📊 GP5 파일 구조

### Song 구조
```
Song
├── title, subtitle, artist, album
├── tempo (Int)
├── measureHeaders[] - 마디 헤더 (박자, 반복 등)
├── tracks[] - 트랙 목록
│   └── Track
│       ├── name
│       ├── strings[] - 튜닝 (GuitarString)
│       ├── offset - 카포 위치
│       ├── fretCount
│       └── measures[] - 마디 목록
│           └── Measure
│               └── voices[] - 2개의 Voice
│                   └── Voice
│                       └── beats[] - 비트 목록
│                           └── Beat
│                               ├── duration
│                               ├── status (normal/rest/empty)
│                               └── notes[]
│                                   └── Note
│                                       ├── string (1-6)
│                                       ├── value (fret 0-24)
│                                       ├── velocity
│                                       └── effect
```

---

## 🎵 Duration 값 매핑

```python
# Duration.value 매핑
DURATION_MAP = {
    -2: "whole",        # 온음표
    -1: "half",         # 2분음표
     0: "quarter",      # 4분음표 (기본값)
     1: "eighth",       # 8분음표
     2: "sixteenth",    # 16분음표
     3: "thirtySecond", # 32분음표
}

# 사용법
beat.duration = gp.Duration(value=0)  # 4분음표
beat.duration = gp.Duration(value=1)  # 8분음표
```

---

## 🎸 Note Effects (테크닉)

### Bend (벤딩)
```python
# BendType 값
# - bend: 벤드 업
# - bendRelease: 벤드 후 릴리즈
# - bendReleaseBend: 벤드 → 릴리즈 → 벤드
# - preBend: 프리벤드
# - preBendRelease: 프리벤드 후 릴리즈
# - dip: 트레몰로 딥

bend = gp.BendEffect()
bend.type = gp.BendType.bend
bend.value = 100  # 반음 = 50, 온음 = 100
bend.points = [
    gp.BendPoint(0, 0),      # 시작
    gp.BendPoint(6, 100),    # 중간 (최고점)
    gp.BendPoint(12, 100)    # 끝
]
note.effect.bend = bend
```

### Slide (슬라이드)
```python
# SlideType 값
# 0x01: shiftSlide - 쉬프트 슬라이드
# 0x02: legatoSlide - 레가토 슬라이드
# 0x04: outDownwards - 아래로 슬라이드 아웃
# 0x08: outUpwards - 위로 슬라이드 아웃
# 0x10: intoFromBelow - 아래에서 슬라이드 인
# 0x20: intoFromAbove - 위에서 슬라이드 인

note.effect.slides = [gp.SlideType.shiftSlide]
```

### Harmonic (하모닉)
```python
# HarmonicType
# 1: natural harmonic (자연 하모닉)
# 2: artificial harmonic (인공 하모닉)
# 3: tapped harmonic (탭 하모닉)
# 4: pinch harmonic (핀치 하모닉)
# 5: semi-harmonic (세미 하모닉)

note.effect.harmonic = gp.NaturalHarmonic()
# 또는
note.effect.harmonic = gp.ArtificialHarmonic(pitch=..., octave=...)
```

### 기타 효과
```python
# Hammer-on / Pull-off
note.effect.hammer = True

# Let Ring
note.effect.letRing = True

# Palm Mute
note.effect.palmMute = True

# Staccato
note.effect.staccato = True

# Vibrato
note.effect.vibrato = True

# Ghost Note
note.effect.ghostNote = True

# Accentuated Note
note.effect.accentuatedNote = True

# Heavy Accentuated
note.effect.heavyAccentuatedNote = True
```

---

## 🎼 Grace Note (꾸밈음)

```python
grace = gp.GraceEffect()
grace.fret = 5  # 프렛 번호
grace.velocity = 95  # 다이나믹
grace.duration = 1  # 1=32분음표, 2=24분음표, 3=16분음표
grace.transition = gp.GraceEffectTransition.none  # none, slide, bend, hammer
grace.isOnBeat = False  # 비트 위인지
grace.isDead = False  # 뮤트된 음인지

note.effect.grace = grace
```

---

## 🔊 Velocity (다이나믹)

```python
# MIDI Velocity 매핑 (GP → MIDI)
VELOCITY_MAP = {
    1: 15,   # ppp
    2: 31,   # pp
    3: 47,   # p
    4: 63,   # mp
    5: 79,   # mf
    6: 95,   # f (기본값)
    7: 111,  # ff
    8: 127   # fff
}

note.velocity = 95  # f (forte)
```

---

## 🎵 Tuplet (연음부)

```python
# N-tuplet 설정
beat.duration.tuplet = gp.Tuplet(3, 2)  # 3연음 (3개 음표를 2개 시간에)
beat.duration.tuplet = gp.Tuplet(5, 4)  # 5연음
beat.duration.tuplet = gp.Tuplet(6, 4)  # 6연음
```

---

## 🎸 Track 설정

### 튜닝 설정
```python
# 표준 튜닝 MIDI 값
# String 1 (High E): 64 (E4)
# String 2: 59 (B3)
# String 3: 55 (G3)
# String 4: 50 (D3)
# String 5: 45 (A2)
# String 6 (Low E): 40 (E2)

track.strings = [
    gp.GuitarString(1, 64),  # 1번줄 = E4
    gp.GuitarString(2, 59),  # 2번줄 = B3
    gp.GuitarString(3, 55),  # 3번줄 = G3
    gp.GuitarString(4, 50),  # 4번줄 = D3
    gp.GuitarString(5, 45),  # 5번줄 = A2
    gp.GuitarString(6, 40),  # 6번줄 = E2
]
```

### 카포 설정
```python
track.offset = 2  # 2프렛에 카포
```

### MIDI 채널 설정
```python
channel = gp.MidiChannel()
channel.channel = 0
channel.effectChannel = 1
channel.instrument = 25  # Steel String Acoustic Guitar
channel.volume = 100
channel.balance = 64  # 중앙
channel.chorus = 0
channel.reverb = 0
channel.phaser = 0
channel.tremolo = 0

track.channel = channel
```

---

## 📝 MeasureHeader 설정

```python
header = gp.MeasureHeader()
header.number = 1
header.start = 960  # Tick 위치
header.tempo = gp.Tempo(120)

# 박자 설정
header.timeSignature = gp.TimeSignature(
    numerator=4,
    denominator=gp.Duration(1)  # 4분음표 기준
)

# 반복 설정
header.repeatOpen = True  # 반복 시작
header.repeatClose = 2  # 반복 끝 (2회 반복)
header.repeatAlternative = 1  # 1번 엔딩

# 마커
header.marker = gp.Marker("Intro", gp.Color(255, 0, 0))
```

---

## 🔧 완전한 GP5 생성 예제

```python
import guitarpro as gp

def create_simple_gp5():
    # 1. Song 생성 (자동으로 Track, Measure 1개씩 생성됨)
    song = gp.Song()
    song.title = "My Song"
    song.artist = "Artist"
    song.tempo = 120
    
    # 2. 기존 Track 수정 (새로 만들지 않음!)
    track = song.tracks[0]
    track.name = "Acoustic Guitar"
    track.fretCount = 24
    track.offset = 0  # 카포 없음
    
    # 3. 튜닝 설정
    track.strings = [
        gp.GuitarString(1, 64),
        gp.GuitarString(2, 59),
        gp.GuitarString(3, 55),
        gp.GuitarString(4, 50),
        gp.GuitarString(5, 45),
        gp.GuitarString(6, 40),
    ]
    
    # 4. 기존 measure 사용
    measure = track.measures[0]
    voice = measure.voices[0]
    
    # 5. Beat 추가
    beat = gp.Beat(voice)
    beat.start = 960
    beat.duration = gp.Duration(value=0)  # Quarter note
    beat.status = gp.BeatStatus.normal
    
    # 6. Note 추가
    note = gp.Note(beat)
    note.string = 1
    note.value = 5  # 5th fret
    note.velocity = 95
    note.type = gp.NoteType.normal
    
    beat.notes.append(note)
    voice.beats.append(beat)
    
    # 7. 저장
    gp.write(song, "output.gp5")
    return song
```

---

## 📚 참고 링크

- **GitHub**: https://github.com/Perlence/PyGuitarPro
- **Documentation**: https://pyguitarpro.readthedocs.io/en/stable/
- **API Reference**: https://pyguitarpro.readthedocs.io/en/stable/#api-reference
- **File Format Spec**: https://pyguitarpro.readthedocs.io/en/stable/#guitar-pro-file-format

---

*이 문서는 OmniTab 프로젝트에서 GP5 파일 생성 시 참조용으로 작성됨*
*마지막 업데이트: 2026-01-18*
