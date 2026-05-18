#!/usr/bin/env python3
"""
Deluge XML → MIDI converter

Handles both clip types:
  • Synth / kit clips  — note data from noteRows
  • MIDI clips         — notes from noteRows + CC automation from midiParams

Outputs:
  {out}/{song_name}/clips/   — one .mid per clip
  {out}/{song_name}/tracks/  — one .mid per instrument/channel, clips concatenated

Note binary format (noteDataWithLift, 11 bytes/note, big-endian):
  [0:4] position ticks  [4:8] length ticks  [8] velocity  [9] lift  [10] flags

CC automation format (midiParams value, big-endian):
  [0:4] initial value (int32)
  then N × 8 bytes: [0:4] value (int32)  [4:8] tick (uint32, top bit = interp flag)
  interp=True means the segment FROM the previous node TO this node is a linear ramp.

Usage: python3 deluge_to_midi.py <song.XML> [output_dir]
Requires: pip install mido
"""

import re
import sys
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

try:
    import mido
except ImportError:
    sys.exit("Missing dependency — install with: pip install mido")

PPQN = 96          # Deluge internal ticks per quarter note
CC_INTERP_STEP = 3 # ticks between generated CC points in interpolated ramps

DRUM_NAME_MAP = {
    "KICK": 36, "BASS": 36, "BD": 36,
    "KIC2": 35, "KIC3": 43, "KIC4": 47,
    "SNAR": 38, "SNR": 38, "SNARE": 38,
    "HATC": 42, "CLOS": 42, "CHH": 42,
    "HATO": 46, "OPEN": 46, "OHH": 46,
    "CLAP": 39, "CLAV": 75,
    "TRIA": 81, "TAMB": 54,
    "TOM1": 45, "TOM2": 47, "TOM3": 50,
    "COWB": 56, "RIDE": 51, "CRSH": 49, "CYMB": 49,
    "REC8": 49, "REC9": 51, "RE11": 56, "RE12": 57,
    "PERC": 60,
}


# ── XML ───────────────────────────────────────────────────────────────────────

def fix_duplicate_attrs(text):
    """Remove duplicate XML attributes (keep first). Deluge emits these on audioClip."""
    def fix_tag(m):
        seen = set()
        def maybe_drop(am):
            name = am.group(1)
            if name in seen:
                return ""
            seen.add(name)
            return am.group(0)
        return re.sub(r'\s+([\w:]+)="[^"]*"', maybe_drop, m.group(0))
    return re.sub(r"<[^>]+>", fix_tag, text, flags=re.DOTALL)

def parse_xml(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return ET.fromstring(fix_duplicate_attrs(text))


# ── Tempo ─────────────────────────────────────────────────────────────────────

def bpm_from_root(root):
    ticks = int(root.get("timePerTimerTick", 248))
    return (44100.0 * 60.0) / (ticks * PPQN)


# ── Shared binary helpers ─────────────────────────────────────────────────────

def _hex_bytes(hex_str):
    s = hex_str[2:] if hex_str.startswith(("0x", "0X")) else hex_str
    return bytes.fromhex(s)

def _is_automated(hex_str):
    """True if the value hex encodes automation (more than a single 4-byte constant)."""
    return len(hex_str) > 10  # "0x" + 8 chars = single value


# ── Note decoding ─────────────────────────────────────────────────────────────

def decode_note_lift(hex_str):
    """noteDataWithLift (11 bytes/note) → [(pos, length, vel), ...]"""
    if not hex_str:
        return []
    raw = _hex_bytes(hex_str)
    out = []
    for i in range(0, len(raw) - 10, 11):
        c = raw[i:i+11]
        out.append((
            struct.unpack(">I", c[0:4])[0],
            struct.unpack(">I", c[4:8])[0],
            max(1, min(127, c[8])),
        ))
    return out

def decode_note_split_prob(hex_str):
    """noteDataWithSplitProb (14 bytes/note) fallback → [(pos, length, vel), ...]"""
    if not hex_str:
        return []
    raw = _hex_bytes(hex_str)
    out = []
    for i in range(0, len(raw) - 13, 14):
        c = raw[i:i+14]
        out.append((
            struct.unpack(">I", c[0:4])[0],
            struct.unpack(">I", c[4:8])[0],
            max(1, min(127, c[8])),
        ))
    return out

def get_notes(row):
    lift = row.get("noteDataWithLift", "")
    if lift:
        return decode_note_lift(lift)
    split = row.get("noteDataWithSplitProb", "")
    if split:
        return decode_note_split_prob(split)
    return []


# ── CC automation decoding ────────────────────────────────────────────────────

def decode_cc_automation(hex_str):
    """
    Returns (initial_val_i32, [(tick, val_i32, interpolated), ...])
    Format: 4-byte header + N × [4-byte value][4-byte tick | interp_flag]
    interp flag = top bit of tick field; means segment from previous node to this is linear.
    """
    raw = _hex_bytes(hex_str)
    if len(raw) < 4:
        return 0, []
    initial = struct.unpack(">i", raw[0:4])[0]
    nodes = []
    for i in range(4, len(raw) - 7, 8):
        val      = struct.unpack(">i", raw[i:i+4])[0]
        tick_raw = struct.unpack(">I", raw[i+4:i+8])[0]
        nodes.append((tick_raw & 0x7FFFFFFF, val, bool(tick_raw & 0x80000000)))
    return initial, nodes

def _i32_to_cc(val):
    """Map Deluge signed int32 to MIDI CC 0-127."""
    return round(((val + 2147483648.0) / 4294967295.0) * 127)

def cc_events_from_automation(initial, nodes):
    """
    Build a list of (tick, cc_value) pairs from decoded automation data.
    Interpolated segments are expanded to CC_INTERP_STEP tick resolution.
    """
    all_pts = [(0, initial, False)] + list(nodes)
    events = [(0, _i32_to_cc(initial))]

    for i in range(1, len(all_pts)):
        prev_tick, prev_val, _ = all_pts[i - 1]
        curr_tick, curr_val, interp = all_pts[i]

        if interp and curr_tick > prev_tick:
            p_cc = _i32_to_cc(prev_val)
            c_cc = _i32_to_cc(curr_val)
            span = curr_tick - prev_tick
            for t in range(prev_tick + CC_INTERP_STEP, curr_tick, CC_INTERP_STEP):
                frac = (t - prev_tick) / span
                events.append((t, round(p_cc + frac * (c_cc - p_cc))))

        events.append((curr_tick, _i32_to_cc(curr_val)))

    return events


# ── Drum mapping ──────────────────────────────────────────────────────────────

def map_drum_name(name):
    upper = name.upper()
    for key, note in DRUM_NAME_MAP.items():
        if key in upper:
            return note
    return 35 + (sum(ord(c) for c in name) % 47)


# ── MIDI file writing ─────────────────────────────────────────────────────────

def make_safe(name):
    return "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_")

def write_midi(path, channel, name, tempo_us, note_events, cc_lanes=None):
    """
    note_events : [(midi_note, pos, length, vel), ...]
    cc_lanes    : {cc_num: [(tick, cc_val), ...]} or None
    """
    mid = mido.MidiFile(type=0, ticks_per_beat=PPQN)
    t = mido.MidiTrack()
    mid.tracks.append(t)
    t.append(mido.MetaMessage("set_tempo",  tempo=tempo_us, time=0))
    t.append(mido.MetaMessage("track_name", name=name,      time=0))
    if channel != 9:
        t.append(mido.Message("program_change", program=0, channel=channel, time=0))

    # Gather all raw events: (abs_tick, sort_priority, mido_message_kwargs)
    raw = []
    for midi_note, pos, length, vel in note_events:
        raw.append((pos,          1, dict(type="note_on",  note=midi_note, velocity=vel, channel=channel)))
        raw.append((pos + length, 0, dict(type="note_off", note=midi_note, velocity=0,   channel=channel)))

    for cc_num, events in (cc_lanes or {}).items():
        for tick, cc_val in events:
            raw.append((tick, 0, dict(type="control_change", control=cc_num, value=cc_val, channel=channel)))

    raw.sort(key=lambda e: (e[0], e[1]))

    cursor = 0
    for abs_tick, _, kwargs in raw:
        msg_type = kwargs.pop("type")
        t.append(mido.Message(msg_type, time=abs_tick - cursor, **kwargs))
        cursor = abs_tick

    mid.save(str(path))


# ── Main conversion ───────────────────────────────────────────────────────────

def convert(xml_path, output_dir=None):
    root = parse_xml(xml_path)

    bpm      = bpm_from_root(root)
    tempo_us = int(60_000_000 / bpm)
    stem     = Path(xml_path).stem

    print(f"Song : {stem}")
    print(f"BPM  : {bpm:.1f}")

    # Output lives in a folder named after the XML file
    if output_dir:
        base = Path(output_dir) / stem
    else:
        base = Path(xml_path).parent / stem

    clips_dir  = base / "clips"
    tracks_dir = base / "tracks"
    clips_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    # Kit pad lists: kit_name → [pad_name, ...]
    kit_pads = {}
    for kit in root.findall(".//kit"):
        name = kit.get("presetName", "KIT")
        kit_pads[name] = [s.get("name", f"PAD{i}") for i, s in enumerate(kit.findall(".//soundSources/sound"))]

    # MIDI channel allocation for synths (ch 9 reserved for drums)
    synth_channels: dict[str, int] = {}
    next_ch = [0]

    def alloc_channel(preset_name):
        if preset_name not in synth_channels:
            if next_ch[0] == 9:
                next_ch[0] += 1
            synth_channels[preset_name] = next_ch[0]
            next_ch[0] += 1
        return synth_channels[preset_name]

    # {track_key: [(section_int, clip_length, channel, note_events, cc_lanes)]}
    track_data: dict[str, list] = defaultdict(list)

    all_clips = root.findall(".//sessionClips/instrumentClip")
    print(f"Clips: {len(all_clips)}\n")

    clip_count = 0

    for idx, clip in enumerate(all_clips):
        preset      = clip.get("instrumentPresetName")
        folder      = clip.get("instrumentPresetFolder", "")
        midi_ch_str = clip.get("midiChannel")
        section     = clip.get("section", str(idx))
        clip_name   = clip.get("clipName", "").strip()
        clip_length = int(clip.get("length", 384))

        is_midi_clip = preset is None and midi_ch_str is not None
        is_kit       = not is_midi_clip and folder.upper() == "KITS"

        # ── Decode notes ────────────────────────────────────────────────────
        note_events = []

        if is_midi_clip:
            channel = int(midi_ch_str)
            for row in clip.findall("noteRows/noteRow"):
                y = int(row.get("y", 60))
                for pos, length, vel in get_notes(row):
                    note_events.append((y, pos, length, vel))

        elif is_kit:
            channel = 9
            pads = kit_pads.get(preset, [])
            for row in clip.findall("noteRows/noteRow"):
                di_str = row.get("drumIndex")
                if di_str is None:
                    continue
                di        = int(di_str)
                pad_name  = pads[di] if di < len(pads) else f"PAD{di}"
                midi_note = map_drum_name(pad_name)
                for pos, length, vel in get_notes(row):
                    note_events.append((midi_note, pos, length, vel))

        else:
            channel = alloc_channel(preset)
            for row in clip.findall("noteRows/noteRow"):
                y = int(row.get("y", 60))
                for pos, length, vel in get_notes(row):
                    note_events.append((y, pos, length, vel))

        # ── Decode CC automation (MIDI clips only) ──────────────────────────
        cc_lanes = {}
        if is_midi_clip:
            midi_params = clip.find("midiParams")
            if midi_params is not None:
                for param in midi_params.findall("param"):
                    cc_el  = param.find("cc")
                    val_el = param.find("value")
                    if cc_el is None or val_el is None:
                        continue
                    cc_num = int(cc_el.text)
                    if cc_num == 255:
                        continue
                    val_hex = (val_el.text or "").strip()
                    if not val_hex or not _is_automated(val_hex):
                        continue
                    initial, nodes = decode_cc_automation(val_hex)
                    events = cc_events_from_automation(initial, nodes)
                    if events:
                        cc_lanes[cc_num] = events

        if not note_events and not cc_lanes:
            continue

        # ── Build clip display name / track key ─────────────────────────────
        if clip_name:
            display = clip_name
        elif is_midi_clip:
            display = f"MIDI_ch{channel + 1}_s{section}"
        else:
            display = f"{preset}_s{section}"

        if is_midi_clip:
            track_key = f"MIDI_ch{channel + 1}"
        elif is_kit:
            track_key = preset
        else:
            track_key = preset

        # ── Write clip file ─────────────────────────────────────────────────
        safe_lbl  = make_safe(clip_name) if clip_name else make_safe(display)
        clip_path = clips_dir / f"{make_safe(stem)}_{safe_lbl}.mid"

        write_midi(clip_path, channel, display, tempo_us, note_events, cc_lanes)

        cc_info = f", {len(cc_lanes)} CC lane(s)" if cc_lanes else ""
        print(f"  clip  {clip_path.name}  ({len(note_events):3d} notes{cc_info}, ch {channel + 1})")
        clip_count += 1

        track_data[track_key].append((int(section), clip_length, channel, note_events, cc_lanes))

    # ── Write per-track files (clips concatenated in section order) ──────────
    print()
    track_count = 0
    for track_key, entries in track_data.items():
        entries.sort(key=lambda e: e[0])
        channel = entries[0][2]

        merged_notes = []
        merged_cc: dict[int, list] = defaultdict(list)
        offset = 0
        for _, clip_length, _, n_evts, cc_evts in entries:
            for midi_note, pos, length, vel in n_evts:
                merged_notes.append((midi_note, pos + offset, length, vel))
            for cc_num, evts in cc_evts.items():
                merged_cc[cc_num].extend((tick + offset, cc_val) for tick, cc_val in evts)
            offset += clip_length

        track_path = tracks_dir / f"{make_safe(stem)}_{make_safe(track_key)}.mid"
        write_midi(track_path, channel, track_key, tempo_us, merged_notes, dict(merged_cc))

        total_notes = sum(len(e[3]) for e in entries)
        total_cc    = sum(len(e[4]) for e in entries)
        cc_info = f", {total_cc} CC lanes" if total_cc else ""
        print(f"  track {track_path.name}  ({len(entries)} clips, {total_notes} notes{cc_info}, ch {channel + 1})")
        track_count += 1

    print(f"\n{clip_count} clip files  → {clips_dir}/")
    print(f"{track_count} track files → {tracks_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <song.XML> [output_dir]")
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
