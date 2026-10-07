# Bộ train NNUE cho Jieqi

Bộ công cụ PyTorch để train mạng NNUE cho engine trong repo này, từ con số 0
(chưa có mạng nào) đến một mạng dùng được, và để kiểm tra các thay đổi bằng
đấu SPRT.

Mạng được train đúng với kiến trúc của engine (`src/nnue`): đặc trưng
HalfKAv2_hm cho cờ úp (7560 đầu vào mỗi bên, có quân úp và pool), feature
transformer 2048 với 16 bucket PSQT, 16 layer stack 2048 → 16 → 32 → 1. Mạng
xuất ra được engine đọc **khớp từng bit** với cách trainer tính (`verify.py`
kiểm tra điều này).

Một phần thiết kế tham khảo bộ train của AB-JChess và nnue-pytorch; code được
viết riêng cho bộ đặc trưng và định dạng file của engine này.

## Mục lục

1. [Các thành phần](#1-các-thành-phần)
2. [Yêu cầu](#2-yêu-cầu)
3. [Cài đặt](#3-cài-đặt)
4. [Quy trình train chi tiết](#4-quy-trình-train-chi-tiết)
5. [Lịch trình gợi ý cho các thế hệ](#5-lịch-trình-gợi-ý-cho-các-thế-hệ)
6. [Dùng engine khác: AB-JChess và nhánh old](#6-dùng-engine-khác-ab-jchess-và-nhánh-old)
7. [Thử nghiệm sau khi có mạng nền](#7-thử-nghiệm-sau-khi-có-mạng-nền)
8. [Định dạng dữ liệu](#8-định-dạng-dữ-liệu)
9. [Dung lượng và thời gian](#9-dung-lượng-và-thời-gian)
10. [Xử lý sự cố](#10-xử-lý-sự-cố)
11. [Khi kiến trúc mạng của engine thay đổi](#11-khi-kiến-trúc-mạng-của-engine-thay-đổi)

## 1. Các thành phần

| File | Công dụng |
| --- | --- |
| `init_net.py` | Tạo mạng khởi đầu chỉ tính vật chất |
| `gen_data.py` | Sinh dữ liệu bằng self-play, danh tính quân úp thật, nhiều tiến trình |
| `prepare_data.py` | Chuyển dữ liệu văn bản thành các shard nhị phân đã xáo trộn (đọc theo luồng, RAM cố định) |
| `train.py` | Vòng lặp train; mỗi epoch ghi một checkpoint và một file `.nnue` |
| `verify.py` | So đánh giá của engine với trainer, phải khớp từng bit |
| `match.py` | Đấu giữa hai mạng / hai bộ tùy chọn / hai bản build, có SPRT |
| `fit_scaling.py` | Fit hằng số điểm → xác suất thắng từ dữ liệu self-play |
| `rescore.py` | Cho một engine khác (ví dụ AB-JChess) chấm điểm lại các thế cờ |
| `serialize.py` | Xuất / đọc file `.nnue`, chuyển `.nnue` thành checkpoint |
| `jieqi_features.py` | Đặc trưng đầu vào, bản sao của `src/nnue/features/half_ka_v2_hm` |
| `model.py` | Mô hình PyTorch và giới hạn trọng số |
| `jieqi_uci.py` | Hàm tiện ích điều khiển engine qua UCI |

Engine có hai thứ phục vụ riêng cho trainer: tùy chọn UCI `RawScore` (in điểm
theo đơn vị nội bộ) và lệnh `evalraw` (in đầu ra thô của mạng).

## 2. Yêu cầu

**Phần cứng**

- **Sinh dữ liệu:** CPU càng nhiều nhân càng tốt, mỗi worker dùng 1 nhân. Có
  thể chạy trên nhiều máy cùng lúc.
- **Train:** GPU NVIDIA có CUDA. CPU vẫn chạy được nhưng chỉ hợp để thử.
  Kiến trúc mặc định cần khoảng 300 MB bộ nhớ GPU cho tham số và optimizer
  (cộng thêm phần cho batch), kiến trúc `att` khoảng 1 GB, kiến trúc
  `threats` khoảng 2 GB.
- **RAM:** khoảng 2 GB cho `prepare_data.py` với shard mặc định; `train.py`
  đọc dữ liệu kiểu memory-map nên không cần nạp hết vào RAM.
- **Ổ đĩa:** nên dùng SSD cho dữ liệu đã chuẩn bị. Xem
  [mục 9](#9-dung-lượng-và-thời-gian) để ước tính dung lượng.

**Phần mềm**

- Trình biên dịch C++17 (g++ hoặc clang) và `make` để build engine.
- Python 3.9 trở lên.
- Các gói trong `requirements.txt`: `torch`, `numpy`, `zstandard`.

## 3. Cài đặt

### 3.1. Build engine

```
cd src
make -j build ARCH=x86-64-avx2
cd ../trainer
```

- Chọn `ARCH` theo CPU của bạn (`make help` liệt kê các lựa chọn, ví dụ
  `x86-64-bmi2`, `x86-64-avx512`, `apple-silicon`). Nếu sinh dữ liệu trên
  nhiều máy khác nhau, build riêng cho từng máy.
- Lần build đầu tiên `make` sẽ tải mạng cờ tướng `master-net` về
  `src/pikafish.nnue` nếu file đó chưa có. Mạng này **không dùng được** cho
  engine cờ úp, nhưng không ảnh hưởng gì: mọi script đều truyền mạng cờ úp
  qua tùy chọn `EvalFile`.

### 3.2. Cài Python

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy zstandard
pip install torch                # xem ghi chú bên dưới
```

Để train trên GPU, cài bản `torch` có CUDA theo hướng dẫn trên
https://pytorch.org (chọn hệ điều hành, `pip`, và phiên bản CUDA của máy).
Kiểm tra:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Phải in ra `True` nếu muốn train bằng GPU. Máy chỉ dùng để sinh dữ liệu thì
không cần GPU, chỉ cần `numpy` và `zstandard` (và `torch` nếu muốn chạy
`verify.py`).

### 3.3. Kiểm tra bộ train khớp với engine

```
python verify.py --engine ../src/pikafish
```

Kết quả phải là `N/N positions bit-exact with the engine`. **Nếu không khớp
thì đừng train**: engine và trainer đang hiểu mạng khác nhau (thường do
engine đã đổi kiến trúc, xem [mục 11](#11-khi-kiến-trúc-mạng-của-engine-thay-đổi)).

### 3.4. Tổ chức thư mục (gợi ý)

```
trainer/
  nets/      mạng đã chọn qua từng thế hệ: gen0.nnue, gen1.nnue, ...
  data/      dữ liệu văn bản (genN.txt, valN.txt) và dữ liệu đã chuẩn bị
  runs/      kết quả train: runs/genN/epoch_K.pt, epoch_K.nnue, last.pt
```

Các thư mục này đã nằm trong `.gitignore`.

## 4. Quy trình train chi tiết

Chưa có mạng cờ úp nào nên phải đi theo vòng lặp, mỗi vòng là một "thế hệ":

```
mạng hiện tại ──► sinh dữ liệu ──► chuẩn bị ──► train ──► kiểm tra ──► đấu SPRT
     ▲                                                                   │
     └────────────── mạng mới thắng thì trở thành mạng hiện tại ◄────────┘
```

Thế hệ 0 bắt đầu từ mạng chỉ tính vật chất. Nếu có AB-JChess, có thể thay thế hệ 0 bằng dữ liệu
do AB sinh ra để mạng đầu tiên mạnh hơn nhiều ([mục 6](#6-dùng-engine-khác-ab-jchess-và-nhánh-old)). Các bước dưới đây mô tả một thế
hệ; lặp lại với số thế hệ tăng dần.

### Bước 1. Mạng khởi đầu (chỉ làm một lần)

```
python init_net.py nets/gen0.nnue
```

Mạng này có trọng số PSQT bằng giá trị quân (quân úp tính bằng giá trị
trung bình của 15 quân một bên), còn các layer stack cho ra 0. Engine sẽ chơi
với đánh giá thuần vật chất cộng với search, đủ để sinh dữ liệu đầu tiên có ý
nghĩa. Lệnh cũng ghi kèm `nets/gen0.nnue.pt` (checkpoint tương ứng).

### Bước 2. Sinh dữ liệu

```
python gen_data.py --engine ../src/pikafish --net nets/gen0.nnue \
    --games 50000 --nodes 5000 --workers 15 --seed 100 --out data/gen0.txt
```

Sinh thêm một **tập kiểm định (validation)** nhỏ, bằng seed khác, để theo dõi
quá trình train:

```
python gen_data.py --engine ../src/pikafish --net nets/gen0.nnue \
    --games 1000 --nodes 5000 --workers 15 --seed 999 --out data/val0.txt
```

Các tham số quan trọng:

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--net` | | Mạng dùng để chơi (mạng hiện tại) |
| `--games` | 100 | Số ván. Mỗi ván cho khoảng 40–120 thế cờ |
| `--nodes` | 5000 | Số node search mỗi nước. Nhiều hơn thì điểm chính xác hơn nhưng chậm hơn tương ứng |
| `--depth` | 0 | Dùng `go depth N` thay cho số node |
| `--workers` | số nhân − 1 | Số tiến trình chạy song song |
| `--seed` | theo giờ hệ thống | **Đặt rõ ràng** và khác nhau cho mỗi lần chạy / mỗi máy |
| `--random-plies` / `--random-plies-min` | 8 / 2 | Số nước ngẫu nhiên đầu ván (không ghi vào dữ liệu), để đa dạng khai cuộc |
| `--max-plies` | 400 | Quá số nửa nước này thì xử hòa |
| `--adjudicate` | 3000 | Xử thắng khi \|điểm\| ≥ ngưỡng này 8 nửa nước liên tiếp |
| `--opt` | | Tùy chọn UCI thêm cho engine, ví dụ `--opt ChanceScaling=400` |

Cần biết thêm:

- Mỗi dòng có dạng `fen | điểm | kết quả`. Điểm tính theo đơn vị nội bộ, của
  bên đi; kết quả là 1 / 0 / −1 (thắng / hòa / thua) của bên đi. Thế cờ đang
  bị chiếu và thế cờ có nước tốt nhất là ăn quân bị bỏ qua.
- Danh tính quân úp được bốc ngẫu nhiên thật cho mỗi ván. Vì chỉ dùng một
  engine cho cả hai bên, cả hai bên đều biết danh tính của quân úp bị ăn.
- Mỗi worker ghi vào một file `.partN` riêng, gộp lại khi tất cả xong. Nếu bị
  ngắt giữa chừng, các file `.partN` vẫn còn và dùng được (mỗi file chỉ chứa
  các ván đã chơi xong).
- **Nhiều máy:** chạy `gen_data.py` trên từng máy với `--seed` khác nhau và
  `--out` khác nhau, rồi chép các file về. Có thể chuẩn bị chung một lần
  (đưa nhiều file vào `prepare_data.py`) hoặc chuẩn bị riêng từng file thành
  các prefix khác nhau; `train.py` nhận nhiều prefix.
- Nên giữ lại các file văn bản: chúng cần để chuẩn bị lại dữ liệu khi đổi
  kiến trúc (ví dụ `att`) và để chạy `fit_scaling.py`. Có thể nén khi lưu trữ,
  nhưng `prepare_data.py` chỉ đọc file văn bản thường, nên phải giải nén trước
  khi dùng.

### Bước 3. Chuẩn bị dữ liệu

```
python prepare_data.py data/gen0.txt --out data/train0
python prepare_data.py data/val0.txt --out data/val0
```

Kết quả là các shard `data/train0-00000.features.npy`, `...meta.npy`, ... và
file `data/train0.arch` ghi kiến trúc.

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--shard-size` | 2 000 000 | Số thế cờ mỗi shard. RAM cần khoảng 2 × 260 byte × giá trị này; giảm nếu thiếu RAM |
| `--max-score` | 10000 | Bỏ thế cờ có \|điểm\| lớn hơn (đơn vị nội bộ, sau khi nhân `--score-scale`) |
| `--score-scale` | 1.0 | Nhân điểm với hệ số này, để đổi điểm của engine khác sang đơn vị nội bộ ([mục 6](#6-dùng-engine-khác-ab-jchess-và-nhánh-old)) |
| `--workers` | số nhân | Số tiến trình phân tích FEN |
| `--arch` | `default` | `att` cho kiến trúc thử nghiệm ([mục 7.2](#72-kiến-trúc-att)) |

Mỗi shard được xáo trộn trước khi ghi; khi train, mỗi batch lấy từ một shard
ngẫu nhiên, nên dữ liệu được trộn đều giữa các shard.

### Bước 4. Train

```
python train.py data/train0 --val data/val0 --out runs/gen1 \
    --epochs 40 --epoch-size 10000000 --batch-size 16384
```

Từ thế hệ thứ hai trở đi, train trên **toàn bộ dữ liệu** của các thế hệ và
bắt đầu từ mạng hiện tại:

```
python train.py data/train0 data/train1 --val data/val1 --out runs/gen2 \
    --init-net nets/gen1.nnue --epochs 40 --epoch-size 10000000
```

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `train` (vị trí) | | Một hoặc nhiều prefix do `prepare_data.py` tạo |
| `--val` | | Một hoặc nhiều prefix kiểm định (dùng tối đa 1 triệu thế cờ) |
| `--out` | | Thư mục kết quả |
| `--epochs` | 100 | **Tổng** số epoch (khi `--resume`, tính cả các epoch đã chạy) |
| `--epoch-size` | 20 000 000 | Số thế cờ mỗi epoch (không bắt buộc bằng kích thước dữ liệu) |
| `--batch-size` | 16384 | Giảm nếu thiếu bộ nhớ GPU |
| `--lr` | 8.75e-4 | Tốc độ học ban đầu |
| `--gamma` | 0.992 | Hệ số giảm tốc độ học sau mỗi epoch |
| `--lambda` | 1.0 | 1.0: chỉ học theo điểm search; nhỏ hơn: trộn thêm kết quả ván |
| `--scaling` | 361 | Hằng số điểm → xác suất ([bước 7](#bước-7-hiệu-chỉnh-hằng-số-điểm--xác-suất)) |
| `--init-net` | | Bắt đầu từ một file `.nnue` (optimizer mới) |
| `--resume` | | Tiếp tục một lần train dở từ checkpoint (cả optimizer và số epoch) |
| `--device` | `cuda` nếu có | `cpu` để ép chạy CPU |
| `--threads` | 0 | Số luồng CPU cho PyTorch (0 = mặc định) |
| `--no-factorize` | | Tắt đặc trưng ảo (mặc định bật, xem dưới) |
| `--factorize-kb` | | Thêm đặc trưng ảo theo bucket vua (AB-JChess dùng; mặc định tắt) |
| `--no-l1-fact` | | Tắt phần fc0 dùng chung cho các layer stack (mặc định bật) |
| `--output-lr-scale` | 0.1 | Tốc độ học của lớp đầu ra, tính theo tỉ lệ với `--lr` |
| `--stack-balance` | 0 (tắt) | Cân bằng loss giữa các layer stack, giá trị là mức trần (ví dụ 8) |
| `--checkpoint-every` | 1 | Chỉ ghi `epoch_K.pt` mỗi K epoch (và ở epoch cuối); `last.pt` và các `.nnue` luôn được ghi |

Hàm mất mát giống nnue-pytorch: điểm của mạng và điểm trong dữ liệu đều được
đổi sang xác suất thắng bằng `sigmoid(điểm / scaling)`, mục tiêu là
`lambda × xác suất từ điểm + (1 − lambda) × kết quả ván`, và loss là trung
bình của `|dự đoán − mục tiêu|^2.5`.

**Các kỹ thuật chỉ dùng khi train** (lấy ý tưởng từ nnue-pytorch và AB-JChess).
Chúng không làm thay đổi định dạng mạng: khi xuất, phần ảo được cộng gộp vào
trọng số thật, nên engine không cần sửa gì và `verify.py` vẫn phải khớp từng bit.

- **Đặc trưng ảo (factorization):** mỗi đặc trưng "quân X ở ô Y trong bucket
  vua B" bật thêm một dòng trọng số dùng chung cho mọi bucket vua của "quân X ở
  ô Y". Điều mạng học được ở một vị trí vua được dùng lại cho các vị trí khác,
  nên mạng học nhanh hơn khi dữ liệu còn ít (lợi càng nhiều với kiến trúc `att`,
  24 bucket). Thử nhỏ trên 2 700 thế cờ: val loss tốt nhất giảm từ 0,00331 xuống
  0,00323. Dòng ảo theo bucket vua (`--factorize-kb`) được cộng một lần cho mỗi
  đặc trưng đang bật nên học rất nhanh; trong cùng bài thử nó làm kết quả kém đi
  (0,00343), nên mặc định tắt.
- **fc0 dùng chung (`l1_fact`):** một phần của lớp fc0 dùng chung cho 16 layer
  stack, để các stack ít mẫu (tàn cuộc) bắt đầu từ những gì các stack nhiều mẫu
  đã học. Giới hạn trọng số áp lên tổng sau khi gộp.
- **Tốc độ học riêng cho lớp đầu ra:** mặc định bằng 1/10 tốc độ chung.
- **Cân bằng layer stack (`--stack-balance CAP`):** mỗi mẫu được nhân trọng số
  `min(CAP, tổng số mẫu / số mẫu của stack đó)`, chuẩn hóa để trung bình mỗi
  stack bằng 1. Với `CAP=8` (giá trị AB dùng) và 16 stack, chỉ các stack chiếm
  hơn 1/8 dữ liệu bị giảm trọng số; muốn nâng các stack hiếm mạnh hơn thì dùng
  CAP lớn hơn (16–32). Khi bật, lệnh in số mẫu và trọng số của từng stack. Val
  loss luôn tính không trọng số (phân bố thật). Đáng thử khi dữ liệu nhiều và
  tàn cuộc quá ít; quyết định bằng `match.py`, không bằng loss.

Hai cái cuối gần như trung tính ở quy mô thử nhỏ; tác dụng của chúng chỉ thấy
được khi dữ liệu lớn. Checkpoint ghi lại cấu hình factorization: `--resume` tự
dùng đúng cấu hình của lần train đó, `--init-net` từ một file `.nnue` thì bắt đầu
với phần ảo bằng 0. Checkpoint cũ (trước khi có các kỹ thuật này) vẫn đọc và
tiếp tục train được; khi đó trạng thái optimizer được tạo mới.

**Theo dõi:** mỗi 100 bước in loss trung bình; cuối mỗi epoch in train loss,
val loss, tốc độ học và thời gian. Val loss nên giảm dần rồi chững lại. Chỉ so
val loss giữa các lần train dùng **cùng một tập val**; loss thấp hơn không có
nghĩa là mạng mạnh hơn, phải đấu mới biết (bước 6).

**Kết quả** trong `runs/gen1/`: mỗi epoch có `epoch_K.pt` (checkpoint, khoảng
190 MB, kiến trúc `att` khoảng 750 MB, `threats` khoảng 1,3 GB) và `epoch_K.nnue` (dùng được ngay
trong engine), cùng `last.pt`. Xóa bớt checkpoint cũ nếu thiếu chỗ, giữ lại
`last.pt` và các file `.nnue`.

**Bị ngắt giữa chừng:** chạy lại đúng lệnh cũ, thêm `--resume runs/gen1/last.pt`.

### Bước 5. Kiểm tra mạng xuất ra

```
python verify.py --engine ../src/pikafish \
    --net runs/gen1/epoch_40.nnue --checkpoint runs/gen1/epoch_40.pt
```

Phải khớp từng bit (`N/N positions bit-exact`). Dòng `largest float/quantized
difference` cho biết sai số do lượng tử hóa giữa mô hình PyTorch và mạng số
nguyên; vài chục đơn vị nội bộ là bình thường.

### Bước 6. Chọn mạng bằng đấu SPRT

Đấu vài epoch cuối với mạng hiện tại:

```
python match.py --engine ../src/pikafish \
    --net-a runs/gen1/epoch_40.nnue --net-b nets/gen0.nnue \
    --nodes 20000 --workers 15 --sprt 0 5
```

- A là bản được kiểm tra, B là bản tham chiếu; kết quả báo theo A.
- Ván được chơi theo cặp: cùng danh tính quân úp, cùng khai cuộc ngẫu nhiên,
  đổi màu. Mỗi engine chỉ biết những gì bên mình được biết (danh tính quân úp
  bị ăn chỉ báo cho bên ăn; `--full-info` để báo cho cả hai).
- `--sprt 0 5`: dừng khi chấp nhận H1 (A mạnh hơn khoảng 5 Elo trở lên) hoặc
  H0 (A không mạnh hơn). Với thay đổi nhỏ dùng `--sprt 0 3`. Khi mạng mới
  chênh lệch lớn (các thế hệ đầu), SPRT kết thúc rất nhanh.
- In định kỳ: số cặp, thắng/hòa/thua của A, thống kê theo cặp (pentanomial),
  Elo ± khoảng tin cậy 95%, và LLR.
- Dùng cùng kiểu giới hạn thời gian với cách engine sẽ chơi thật (`--nodes`,
  `--depth` hoặc `--movetime`). Mỗi engine dùng 1 luồng; `--workers` cặp ván
  chạy song song.

Nếu mạng mới thắng (H1), chép nó thành mạng hiện tại:

```
cp runs/gen1/epoch_40.nnue nets/gen1.nnue
```

Nếu thua (H0), thử epoch khác, thêm dữ liệu, hoặc chỉnh tham số train
(`--lr`, `--lambda`, số epoch) rồi train lại.

### Bước 7. Hiệu chỉnh hằng số điểm → xác suất

Sau khi có mạng mới, sinh dữ liệu bằng mạng đó (bước 2 của thế hệ sau) rồi:

```
python fit_scaling.py data/gen1.txt
```

Script in hằng số phù hợp nhất và một bảng so sánh tỉ lệ thắng quan sát được
với tỉ lệ dự đoán. Dùng giá trị đó cho:

- engine khi sinh dữ liệu và khi đấu: `--opt ChanceScaling=<s>` (cho
  `gen_data.py`) / `--opt-a ChanceScaling=<s> --opt-b ChanceScaling=<s>`
  (cho `match.py`), hoặc `setoption name ChanceScaling value <s>` trong GUI;
- trainer: `train.py --scaling <s>`.

Fit lại sau mỗi thế hệ. Nên kiểm tra bằng `match.py` rằng giá trị mới không
làm engine yếu đi trước khi dùng mặc định.

### Bước 8. Thế hệ tiếp theo

1. Sinh dữ liệu bằng mạng hiện tại (`nets/genN.nnue`) với seed mới, cùng một
   tập val mới (`valN.txt`).
2. Chuẩn bị thành `data/trainN` và `data/valN`.
3. Train trên tất cả dữ liệu: `train.py data/train0 ... data/trainN --val data/valN --init-net nets/genN.nnue --out runs/genN+1`.
4. Kiểm tra, đấu với `nets/genN.nnue`, giữ mạng thắng.

Khi đã có nhiều thế hệ, có thể bỏ dữ liệu của các thế hệ đầu (chất lượng
thấp hơn) khỏi lệnh train.

## 5. Lịch trình gợi ý cho các thế hệ

Bảng dưới đây chỉ là **điểm khởi đầu**, chưa được kiểm chứng trên quy mô
thật; hãy điều chỉnh theo kết quả đấu.

| Thế hệ | Mạng sinh dữ liệu | Dữ liệu mới | `--nodes` | Train | `--lambda` |
| --- | --- | --- | --- | --- | --- |
| 1 | gen0 (vật chất) | 5–10 triệu thế cờ | 5000 | 20–30 epoch × 10 triệu | 1.0 |
| 2 | gen1 | 10–20 triệu | 5000 | 30–40 epoch × 10 triệu, `--init-net` | 1.0 |
| 3 | gen2 | 20–40 triệu | 5000–10000 | 40–60 epoch × 20 triệu | 1.0 |
| 4+ | genN | 50 triệu trở lên | 10000 | 60–100 epoch × 20–50 triệu | 0.75–1.0 |

Nguyên tắc chung:

- Các thế hệ đầu cải thiện nhanh; tăng dữ liệu và số node dần khi mức cải
  thiện chậm lại.
- Chỉ giảm `--lambda` (trộn kết quả ván) khi mạng đã khá, vì kết quả của các
  ván chơi bởi mạng yếu rất nhiễu.
- Sau vài thế hệ, thử train lại từ đầu (không `--init-net`) trên toàn bộ dữ
  liệu tốt và so với mạng train tiếp; cách nào thắng thì giữ.

## 6. Dùng engine khác: AB-JChess và nhánh old

Hai engine cờ úp hoàn chỉnh khác dùng được với các công cụ ở đây (đã kiểm
tra: lệnh `d`, `go perft`, cách viết nước đi có danh tính lật, nước ăn quân
úp không kèm danh tính đều tương thích):

- **AB-JChess** (https://github.com/lxsgx23/AB-JChess): có mạng NNUE đã train,
  phát hành trong `Releases` (file `0.2b_bmi2.zip` gồm `AB-JChess.exe` cho
  Windows và mạng `abjchess-20260911.nnue`). Mạnh hơn nhiều so với mạng chỉ
  tính vật chất, nên dùng làm **"thầy"** để sinh dữ liệu (chưng cất) và làm
  **thước đo**.
- **Nhánh `old`** của repo này: đánh giá thủ công, chưa đo sức mạnh, và đánh
  giá **lệch màu** (khoảng +375 cp cho Đỏ ở thế khai cuộc với 5000 node). Không
  nên dùng làm thầy (mạng sẽ học cả độ lệch); dùng làm **đối thủ tham chiếu**.

Hai engine này không có tùy chọn `RawScore`: điểm in ra là cp đã chuẩn hóa
theo thang riêng của từng engine. Vì vậy dữ liệu do chúng chấm điểm phải
được **đổi thang** trước khi train (bước 6.3).

### 6.1. Build AB-JChess

```
git clone https://github.com/lxsgx23/AB-JChess
# tải 0.2b_bmi2.zip từ trang Releases, chép abjchess-20260911.nnue vào AB-JChess/src
make -C AB-JChess/src -j build ARCH=x86-64-avx2
```

Trên Windows có thể dùng thẳng `AB-JChess.exe` trong file zip. Khi gọi AB, nên
thêm `ChasingRule=github` để AB dùng luật bắt dai gần với engine này (mặc định
của AB là SkyRule).

### 6.2. Cách A: AB tự chơi để sinh dữ liệu

Thay cho bước 1–2 của thế hệ đầu (mạng vật chất):

```
python gen_data.py --engine AB-JChess/src/AB-JChess \
    --net AB-JChess/src/abjchess-20260911.nnue --opt ChasingRule=github \
    --games 50000 --nodes 5000 --adjudicate 1000 --workers 15 \
    --seed 100 --out data/ab0.txt
```

- `--adjudicate` tính theo thang cp của AB. 1000 chỉ là giá trị ban đầu: sau
  khi có dữ liệu, xem bảng của `fit_scaling.py` (bước 6.3) và chọn mức điểm mà
  tỉ lệ thắng quan sát được gần 100%.
- AB chậm hơn engine này (mạng lớn hơn nhiều): đo sơ bộ khoảng 7 thế cờ/giây
  mỗi worker trong một lần chạy ngắn (gồm cả thời gian nạp mạng).

### 6.3. Đổi thang điểm và train

```
python fit_scaling.py data/ab0.txt
# giả sử in ra "fitted scale: 250.0" -> hệ số = 361 / 250 = 1.444
python prepare_data.py data/ab0.txt --out data/train_ab0 --score-scale 1.444
python prepare_data.py data/ab_val.txt --out data/val_ab --score-scale 1.444
python train.py data/train_ab0 --val data/val_ab --out runs/ab1 \
    --epochs 40 --epoch-size 10000000
```

`361` là hằng số mặc định của trainer (`--scaling`) và gần bằng hằng số engine
này dùng; nếu đã fit được hằng số riêng cho engine này thì dùng nó thay cho
361. Hệ số đổi thang cần fit lại cho mỗi thiết lập của AB (mạng, số node).

Dữ liệu đã đổi thang dùng chung được với dữ liệu tự chơi của engine này:
chuẩn bị riêng từng phần (mỗi phần với `--score-scale` của nó) rồi đưa nhiều
prefix vào `train.py`.

### 6.4. Cách B: chấm điểm lại dữ liệu tự chơi bằng AB

Khi engine này đã có mạng, dữ liệu tự chơi của nó đa dạng và rẻ để sinh, còn
AB chấm điểm chính xác hơn. Kết hợp hai thứ bằng `rescore.py`:

```
python rescore.py data/gen3.txt --out data/gen3_ab.txt \
    --engine AB-JChess/src/AB-JChess --net AB-JChess/src/abjchess-20260911.nnue \
    --opt ChasingRule=github --nodes 20000 --workers 15
python fit_scaling.py data/gen3_ab.txt
python prepare_data.py data/gen3_ab.txt --out data/train3_ab --score-scale <361/s>
```

- Mỗi thế cờ giữ kết quả ván gốc và nhận điểm search của AB cho bên đi.
- Thế cờ AB báo chiếu bí, hoặc có nước tốt nhất là ăn quân, bị bỏ.
- Bị ngắt thì chạy lại đúng lệnh, thêm `--resume`.

### 6.5. Đo sức mạnh với AB và nhánh old

```
# so với AB (dùng --movetime: số node của hai engine không so sánh được)
python match.py --engine-a ../src/pikafish --net-a nets/genN.nnue \
    --engine-b AB-JChess/src/AB-JChess --net-b AB-JChess/src/abjchess-20260911.nnue \
    --opt-b ChasingRule=github --movetime 1000 --workers 15

# so với nhánh old (không cần --net-b)
python match.py --engine-a ../src/pikafish --net-a nets/genN.nnue \
    --engine-b ../../jieqi-old/src/PikaJieQi --movetime 1000 --workers 15
```

Không cần `--sprt` khi chỉ muốn ước lượng Elo: bỏ đi và đặt `--pairs`. Vì thang
điểm của hai bên khác nhau, ngưỡng xử thắng của `match.py` hầu như không kích
hoạt; ván kết thúc bằng chiếu bí, 120 nửa nước không ăn quân hoặc
`--max-plies` (chậm hơn nhưng vẫn đúng).

Lộ trình gợi ý: train vài thế hệ đầu bằng dữ liệu của AB (cách A), sau đó tự
chơi bằng engine này và chấm điểm lại bằng AB (cách B), và chuyển hẳn sang tự
chơi khi mạng của engine này tiến gần AB trong các trận đo.

### 6.6. Về quyền sử dụng

Mã nguồn của AB-JChess theo GPL-3; bản phát hành ghi miễn phí và cấm bán lại.
Một mạng train từ điểm đánh giá của AB có thể coi là sản phẩm phái sinh: hãy
ghi công AB-JChess khi phát hành mạng và cân nhắc hỏi ý các tác giả.

## 7. Thử nghiệm sau khi có mạng nền

### 7.1. Các tùy chọn search

Engine có các tùy chọn thử nghiệm (xem README chính), kiểm tra từng cái một
bằng `match.py`, cùng một mạng:

```
python match.py --engine ../src/pikafish --net-a nets/genN.nnue \
    --opt-a TTPenalize=true --nodes 20000 --workers 15 --sprt 0 3
```

Các tùy chọn đáng thử: `ProbCutDarkCapture`, `TTPenalize`,
`ChanceBranchDepth`, `JieqiMaterial` (bật/tắt, mặc định tắt);
`ChanceRiskAversion` (ví dụ 10, 20, −10); `HiddenCaptureModel=false` và
`RevealQuietBase=0 RevealQuietSafety=0 RevealQuietHighValue=0` (để kiểm tra
lại hai tính năng đang bật mặc định). Tùy chọn nào thắng SPRT thì có thể đặt
làm mặc định trong `src/engine.cpp`.

### 7.2. Kiến trúc `att`

Kiến trúc thử nghiệm có bucket tấn công và mid mirror (xem README chính).
Mạng `att` không tương thích với bản build thường.

**Build engine riêng** (phải `make clean` khi đổi giữa hai kiến trúc, vì các
file object không được build lại tự động):

```
cd src
make clean && make -j build ARCH=x86-64-avx2 nnue=att && mv pikafish pikafish-att
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/pikafish-att --arch att
```

**Train** trên cùng dữ liệu văn bản, chuẩn bị lại với `--arch att`:

```
python init_net.py nets/att0.nnue --arch att          # hoặc bỏ qua, xem dưới
python prepare_data.py data/gen0.txt data/gen1.txt --out data/train_att --arch att
python prepare_data.py data/val1.txt --out data/val_att --arch att
python train.py data/train_att --val data/val_att --out runs/att1 \
    --epochs 40 --epoch-size 10000000
python verify.py --engine ../src/pikafish-att --net runs/att1/epoch_40.nnue
```

`train.py` tự lấy kiến trúc từ dữ liệu; `--init-net`, `--resume`,
`verify.py` và `serialize.py` tự nhận kiến trúc từ file và báo lỗi nếu không
khớp. Không cần `init_net.py` trừ khi muốn sinh dữ liệu bằng engine `att` từ
đầu: dữ liệu văn bản từ engine thường dùng chung được.

**So sánh** với mạng thường train trên **cùng dữ liệu** (công bằng):

```
python match.py --engine-a ../src/pikafish-att --net-a runs/att1/epoch_40.nnue \
    --engine-b ../src/pikafish --net-b nets/genN.nnue \
    --nodes 20000 --workers 15 --sprt 0 5
```

Mạng `att` có thể chậm hơn (accumulator phải tính lại nhiều hơn), nên so
bằng `--movetime` sẽ công bằng hơn `--nodes`.

### 7.3. Kiến trúc `threats` (đặc trưng đe dọa)

Ý tưởng "full threats" của Stockfish / Pikafish, chuyển sang Jieqi: ngoài các
đặc trưng HalfKA, mạng có thêm một đặc trưng cho mỗi cặp (quân tấn công, ô đi,
ô đến, quân bị tấn công) đang có trên bàn cờ, kể cả bảo vệ quân mình:

- quân chia làm 16 lớp: mỗi loại quân đã lật của mỗi bên, cộng "quân úp" của
  mỗi bên; quân úp tấn công như loại quân của ô xuất phát của nó (sĩ úp chỉ
  trong cung);
- tổng cộng 94 037 đặc trưng, thường 10–40 đặc trưng bật cùng lúc mỗi bên;
- feature transformer còn 1024 đầu ra (thay vì 2048) để bù chi phí.

Mạng `threats` (và `att-threats` = `att` + đe dọa) **không tương thích** với
các bản build khác. Build riêng (nhớ `make clean`):

```
cd src
make clean && make -j build ARCH=x86-64-avx2 threats=yes && mv pikafish pikafish-thr
make clean && make -j build ARCH=x86-64-avx2 threats=yes nnue=att && mv pikafish pikafish-att-thr
make clean && make -j build ARCH=x86-64-avx2
cd ../trainer
python verify.py --engine ../src/pikafish-thr --arch threats
python verify.py --engine ../src/pikafish-att-thr --arch att-threats
```

**Train** giống `att`: cùng dữ liệu văn bản, chuẩn bị lại với `--arch threats`
(dữ liệu đã chuẩn bị của các kiến trúc khác không dùng được):

```
python init_net.py nets/thr0.nnue --arch threats      # chỉ khi cần sinh dữ liệu từ đầu
python prepare_data.py data/gen0.txt data/gen1.txt --out data/train_thr --arch threats
python prepare_data.py data/val1.txt --out data/val_thr --arch threats
python train.py data/train_thr --val data/val_thr --out runs/thr1 \
    --epochs 40 --epoch-size 10000000 --checkpoint-every 10
python verify.py --engine ../src/pikafish-thr --net runs/thr1/epoch_40.nnue \
    --checkpoint runs/thr1/epoch_40.pt
```

Trọng số đe dọa bắt đầu từ 0 (mạng mới tính giống hệt khi không có đe dọa) và
được giữ trong khoảng của int8 khi train. `--init-net` chỉ nhận mạng cùng kiến
trúc; không chuyển được mạng `default` (2048 đầu ra) sang `threats`.

**Chi phí**, cần cân nhắc khi so sánh:

| | `default` | `threats` |
| --- | --- | --- |
| Tốc độ engine (bench, 1 luồng, AVX2) | 100 % | khoảng 65–70 % |
| Bộ nhớ engine cho mạng | khoảng 32 MB | khoảng 120 MB (trọng số đe dọa int8 ≈ 96 MB) |
| File `.nnue` (nén zstd) | khoảng 8 MB | tới khoảng 60 MB khi trọng số đã train |
| Dữ liệu đã chuẩn bị | khoảng 260 byte/thế cờ | khoảng 350 byte/thế cờ |
| `prepare_data.py` | khoảng 15 000 dòng/giây mỗi nhân | khoảng 6 000 dòng/giây mỗi nhân |
| Checkpoint | khoảng 190 MB | khoảng 1,3 GB |

Vì engine chậm hơn khoảng một phần ba, hãy so với mạng thường train trên cùng
dữ liệu bằng `--movetime` (hoặc ít nhất kiểm tra lại bằng `--movetime` sau khi
thắng với `--nodes`):

```
python match.py --engine-a ../src/pikafish-thr --net-a runs/thr1/epoch_40.nnue \
    --engine-b ../src/pikafish --net-b nets/genN.nnue \
    --movetime 100 --workers 15 --sprt 0 5
```

## 8. Định dạng dữ liệu

**Văn bản** (`gen_data.py` ghi ra, `prepare_data.py` và `fit_scaling.py` đọc):

```
xxxxkxxxx/9/1x5x1/x1x1x3x/1P4n2/9/X1X1X1X1X/6B2/9/XXXXKXXXX b R2A2C2P4N2B1r2a2c2p5n1b2 5 3 | -272 | 1
```

FEN của engine (có pool), điểm của bên đi theo đơn vị nội bộ, kết quả của
bên đi (1 / 0 / −1). Có thể ghép dữ liệu từ nguồn khác nếu đúng định dạng này
và điểm theo đơn vị nội bộ (`RawScore`).

**Đã chuẩn bị** (mỗi shard, khoảng 260 byte một thế cờ):

- `PREFIX-NNNNN.features.npy`: `int16 (N, 2, 64)`, chỉ số đặc trưng của bên đi
  và bên kia, phần thừa điền bằng chỉ số đệm;
- `PREFIX-NNNNN.meta.npy`: bản ghi `bucket` (layer stack), `result`, `score`;
- `PREFIX.arch`: tên kiến trúc.

Với kiến trúc `threats` / `att-threats`, mỗi shard có thêm (khoảng 90 byte một
thế cờ):

- `PREFIX-NNNNN.threats.npy`: `uint32 (T,)`, chỉ số đặc trưng đe dọa của mọi
  thế cờ nối liền nhau, mỗi thế cờ gồm phần của bên đi rồi của bên kia;
- `PREFIX-NNNNN.thrcount.npy`: `uint8 (N, 2)`, số đặc trưng đe dọa của bên đi
  và của bên kia.

## 9. Dung lượng và thời gian

Các con số đo trên máy thử, chỉ để ước lượng; hãy đo lại trên máy của bạn.

| Mục | Ước tính |
| --- | --- |
| Tốc độ sinh dữ liệu | khoảng 40–50 thế cờ/giây mỗi worker với `--nodes 5000` (gần tỉ lệ nghịch với số node) |
| Máy 16 nhân, `--nodes 5000` | khoảng 2–2,5 triệu thế cờ/giờ |
| File văn bản | khoảng 100 byte/thế cờ (10 triệu ≈ 1 GB) |
| Dữ liệu đã chuẩn bị | khoảng 260 byte/thế cờ (10 triệu ≈ 2,6 GB) |
| `prepare_data.py` | khoảng 15 000 dòng/giây mỗi nhân CPU |
| Checkpoint | khoảng 190 MB (`att`: 750 MB, `threats`: 1,3 GB) mỗi epoch |
| File `.nnue` | khoảng 8 MB (`att`: lớn hơn, `threats`: tới khoảng 60 MB), nén zstd |

Tốc độ train phụ thuộc GPU; hãy chạy thử một epoch nhỏ (`--epoch-size
1000000`) để đo trước khi đặt lịch.

## 10. Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách xử lý |
| --- | --- |
| `verify.py` báo `MISMATCH` | Engine và trainer không cùng kiến trúc. Build lại engine từ cùng commit với trainer; với `att` / `threats` nhớ `make clean` và `--arch` tương ứng |
| Engine báo `network file ... was not loaded` | Sai kiến trúc (mạng `att` / `threats` trên engine thường hoặc ngược lại; `threats=yes` và `nnue=att` phải khớp với kiến trúc của mạng), hoặc file không phải `.nnue` nén zstd do trainer xuất |
| `no threat data (prepare the data again ...)` | Dữ liệu được chuẩn bị bằng phiên bản cũ của `prepare_data.py`; chạy lại với `--arch threats` |
| `incompatible network for arch ...` | Truyền mạng sai kiến trúc cho `--init-net` / `serialize.py`; thông báo cho biết kiến trúc thật của file |
| `prepare_data.py` hết RAM | Giảm `--shard-size` (ví dụ 500000) |
| `train.py` hết bộ nhớ GPU | Giảm `--batch-size` (ví dụ 8192); kiến trúc `att` cần nhiều hơn |
| Loss thành `nan` hoặc tăng vọt | Giảm `--lr` (ví dụ 5e-4), kiểm tra dữ liệu có điểm bất thường (giảm `--max-score`); thử `--no-factorize` để xem factorization có phải nguyên nhân |
| `warning: optimizer state not restored` khi `--resume` | Checkpoint tạo bởi phiên bản trainer cũ (một nhóm tham số): mô hình vẫn được nạp, chỉ trạng thái optimizer được tạo mới |
| Train chậm, GPU không bận | Dữ liệu nằm trên ổ chậm: chuyển sang SSD; tăng `--batch-size` |
| `gen_data.py` bị ngắt | Dùng các file `.partN` còn lại như dữ liệu bình thường |
| AB-JChess / nhánh old in `No such option: RawScore` | Bình thường: các engine này không có tùy chọn đó; điểm của chúng phải đổi thang bằng `--score-scale` ([mục 6.3](#63-đổi-thang-điểm-và-train)) |
| Hai lần sinh dữ liệu trùng nhau | Quên đặt `--seed` khác nhau |
| SPRT mãi không kết thúc | Hai bên gần như ngang nhau; dừng tay và coi là không cải thiện, hoặc dùng khoảng hẹp hơn như `--sprt 0 3` với số cặp lớn |

## 11. Khi kiến trúc mạng của engine thay đổi

`jieqi_features.py` (các bộ đặc trưng, `ARCHS`), `model.py` và các hằng số
đầu file `serialize.py` (hash, kích thước, hệ số lượng tử) mô tả engine. Sau
mọi thay đổi trong `src/nnue`, cập nhật các file này và chạy `verify.py` cho
từng kiến trúc cho đến khi khớp từng bit. Dữ liệu đã chuẩn bị phải được tạo
lại; dữ liệu văn bản vẫn dùng được.

Với các kiến trúc có đe dọa, phần "Threat features" của `jieqi_features.py`
phải liệt kê đặc trưng theo đúng thứ tự của `src/nnue/features/jieqi_threats.cpp`
(engine kiểm tra số đặc trưng lúc khởi động và dừng nếu khác 94 037).
