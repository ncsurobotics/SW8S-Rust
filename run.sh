cd ~/deploy/
while true; do
  case $1 in
    1)
      ln -f -s config_1.toml config.toml
      ./sw8s_rust arm gate_run_reckon spin slalom_left slalom_left slalom_left static_align
    ;;
    2)
      ln -f -s config_2.toml config.toml
      ./sw8s_rust arm gate_run_reckon spin slalom_left slalom_right slalom_right static_align
    ;;
    3)
      ln -f -s config_3.toml config.toml
      ./sw8s_rust arm gate_run_reckon spin
    ;;
    *) echo "NO"
  esac
done
