.PHONY: build run release run-release install clean

build:
	cargo build

run:
	cargo run

release:
	cargo build --release
	@mkdir -p ~/.local/bin
	@cp -f target/release/commandcode2api ~/.local/bin/c2c
	@echo "→ Đã copy sang ~/.local/bin/c2c"

run-release:
	cargo run --release
	@mkdir -p ~/.local/bin
	@cp -f target/release/commandcode2api ~/.local/bin/c2c
	@echo "→ Đã copy sang ~/.local/bin/c2c"

install: release

clean:
	cargo clean
