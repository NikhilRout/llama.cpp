# Vortex llama bitstream targets (mirror ../CMakeLists.txt): H = NT16 2xDXA 128B/cyc 64KB,
# L1 = NT8 FEDP2K 16KB, L2 = NT16 16KB. All: 1 core, IW=4, NW=16, WGMMA+DXA, FP16/BF16+NVFP4+LNSF4.
VX_BITSTREAMS=${VX_BITSTREAMS:-$HOME/vortex_bitstreams}
VX_HOME=${VX_HOME:-$HOME/vortex}
vx_bitstream() {
  case $1 in
    H)  echo llama_heavy_nt16_dxa2 ;;
    L1) echo llama_lite_nt8_f2k_cong ;;
    L2) echo llama_lite_nt16 ;;
    *)  echo "unknown target $1" >&2; return 1 ;;
  esac
}
