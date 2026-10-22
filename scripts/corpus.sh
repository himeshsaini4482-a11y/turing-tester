# Stage 1, sourced by run_benchmark.sh and run_sanity_check.sh (needs CORPUS and TTS_PY).
# Sets MANIFEST (unless given) and makes sure the locked set and its audio exist.
case "$CORPUS" in
  ears)
    MANIFEST=${MANIFEST:-benchmark/manifests/ears_regular_26x19.csv}
    echo "== Stage 1: EARS benchmark set ($MANIFEST)"
    # creates the manifest once; afterwards only re-downloads audio missing on this machine
    $TTS_PY benchmark/prepare_ears.py --speakers 26 --out "$MANIFEST" --limit "${LIMIT:-0}"
    ;;
  libritts-r)
    MANIFEST=${MANIFEST:-benchmark/manifests/libritts_r_test_clean_10x50.csv}
    if [ ! -f "$MANIFEST" ]; then
      echo "== Stage 1: selecting LibriTTS-R benchmark sentences"
      $TTS_PY benchmark/prepare_data.py --corpus libritts-r --speakers 10 --sentences-per-speaker 50 --out "$MANIFEST"
    fi
    ;;
  *) echo "unknown CORPUS=$CORPUS (ears | libritts-r)" >&2; exit 1 ;;
esac
