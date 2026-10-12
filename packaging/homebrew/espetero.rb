# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

# Homebrew formula: builds Espetero from the source tarball of a release.
#
# The tap (github.com/JarJarFran/homebrew-espetero) carries this formula with
# the sha256 of the release's source tarball:
#   curl -sL https://github.com/JarJarFran/espetero/archive/refs/tags/v0.1.1.tar.gz | shasum -a 256
class Espetero < Formula
  desc "Fast LLM inference for Apple silicon with OpenAI and Anthropic APIs"
  homepage "https://github.com/JarJarFran/espetero"
  url "https://github.com/JarJarFran/espetero/archive/refs/tags/v0.1.1.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000" # set in the tap (see above)
  license all_of: ["MIT", "Apache-2.0"] # Apache-2.0: src/engine/clef.{h,cpp}

  depends_on arch: :arm64
  depends_on :macos

  def install
    system "make", "-j#{ENV.make_jobs}"
    bin.install "build/espetero"
  end

  def caveats
    <<~EOS
      `brew services start espetero` runs `espetero serve` at login with the
      settings of ~/.config/espetero/config.json, which must name the model:
        {"model": "mlx-community/Qwen3.8-27B-4bit"}
      (`espetero service install --model ...` sets up the same without Homebrew;
      run one of the two, they would share the port.)
    EOS
  end

  # brew services: at login, restarted if it fails (not after a clean stop);
  # without a process type launchd would throttle the server's CPU and I/O.
  service do
    run [opt_bin/"espetero", "serve"]
    keep_alive successful_exit: false
    process_type :interactive
    environment_variables PATH: std_service_path_env
    log_path var/"log/espetero.log"
    error_log_path var/"log/espetero.log"
  end

  test do
    assert_match "espetero #{version}", shell_output("#{bin}/espetero --version")
  end
end
