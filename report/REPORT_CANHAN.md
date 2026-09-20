# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Thanh Trường
**MSSV:** 2A202602492
**Nhóm:** *(bổ sung sau)*
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**

Hai đoạn văn bản có cosine cao nghĩa là vector nhúng của chúng **gần như cùng một hướng trong không gian** — tức mô hình cho rằng chúng nói về cùng một chủ đề, cùng một nội dung. Điểm này không phụ thuộc độ dài văn bản: một câu ngắn và một đoạn dài cùng nội dung vẫn có cosine cao, vì cosine chỉ đo **góc**, không đo **độ lớn**.

**Ví dụ có độ tương tự CAO:**

- Câu A: *"Người mua có thể gửi yêu cầu trả hàng trong vòng 15 ngày kể từ khi nhận hàng."*
- Câu B: *"Thời hạn để gửi yêu cầu hoàn tiền là mười lăm ngày sau khi nhận được sản phẩm."*
- Điểm đo được: **+0.8428**
- Tại sao tương đồng: hai câu cùng nói về một thời hạn gửi yêu cầu, chỉ khác cách viết con số (`15` so với `mười lăm`) và khác động từ (`trả hàng` so với `hoàn tiền`). Đây là cặp được thiết kế để kiểm chứng embedding hiểu **nghĩa** chứ không chỉ so khớp mặt chữ: nếu mô hình chỉ đếm từ trùng, hai câu này đã không thể đạt 0.84.

**Ví dụ có độ tương tự THẤP:**

- Câu A: *"Hàng cồng kềnh là đơn hàng có khối lượng từ 10kg trở lên."*
- Câu B: *"Hôm nay trời Hà Nội mát mẻ và có mưa nhẹ vào buổi chiều."*
- Điểm đo được: **−0.0162**
- Tại sao khác: hai câu không chia sẻ chủ đề nào — một câu nói về quy cách vận chuyển, một câu nói về thời tiết. Điểm gần 0 (thậm chí hơi âm) là dấu hiệu đúng: hai vector gần như **vuông góc**, tức không liên quan.

**Tại sao độ tương tự cosine được ưu tiên hơn khoảng cách Euclid cho text embeddings?**

Vì độ dài văn bản là thông tin **không liên quan đến ngữ nghĩa** nhưng lại ảnh hưởng mạnh tới khoảng cách Euclid. Một đoạn văn dài gấp đôi về cùng một chủ đề sẽ nằm xa hơn theo Euclid, dù nghĩa y hệt. Cosine bỏ qua độ lớn vector nên loại bỏ được nhiễu đó. Ngoài ra, các mô hình nhúng văn bản đều **chuẩn hoá vector về độ dài 1** (`||v|| = 1`), khi đó cosine bằng đúng tích vô hướng — đó cũng là lý do `EmbeddingStore._search_records` dùng `_dot` để tính điểm thay vì tự chia chuẩn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Công thức: `ceil((độ_dài − overlap) / (chunk_size − overlap))`

```
ceil((10000 − 50) / (500 − 50))
= ceil(9950 / 450)
= ceil(22.11)
= 23 chunks
```

**Đáp án: 23 chunks.**

Không tin công thức suông, tôi đã kiểm lại bằng chính `FixedSizeChunker` có sẵn trong repo:

```python
len(FixedSizeChunker(chunk_size=500, overlap=50).chunk("a" * 10000))   # → 23
```

**Nếu overlap tăng lên 100, số chunk thay đổi thế nào?**

```
ceil((10000 − 100) / (500 − 100))
= ceil(9900 / 400)
= ceil(24.75)
= 25 chunks
```

Kiểm chứng bằng `FixedSizeChunker(chunk_size=500, overlap=100)` → cũng ra **25**. Tăng 2 chunk.

**Tại sao lại muốn tăng độ chồng chéo?**

Vì overlap là **bảo hiểm chống mất ngữ cảnh ở đường cắt**. Bước nhảy giữa hai chunk giảm từ 450 xuống 400 ký tự, nên một câu hoặc một ý nằm vắt qua ranh giới sẽ xuất hiện **trọn vẹn** trong ít nhất một chunk. Đánh đổi là tốn thêm dung lượng lưu trữ và thời gian nhúng, cùng nguy cơ một chunk trùng lặp chiếm mất slot trong top-k. Với văn bản quy định — nơi một điều kiện và ngoại lệ của nó có thể nằm ở hai câu liền nhau — tôi cho rằng overlap cao là đáng, vì mất một ngoại lệ nguy hiểm hơn nhiều so với việc trả về một chunk hơi trùng.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:

Cái bẫy nằm ở biểu thức chính quy. Cách viết ngây thơ `re.split(r"[.!?]\s+", text)` sẽ **nuốt mất dấu câu**, biến mọi chunk thành câu cụt. Tôi dùng lookbehind để cắt ở vị trí **sau** dấu câu mà vẫn giữ nó lại:

```python
re.split(r"(?<=[.!?])\s+", text.strip())
```

Một biểu thức này xử lý luôn cả bốn trường hợp mà docstring yêu cầu (`". "`, `"! "`, `"? "`, `".\n"`), vì `\s+` bao trùm cả dấu cách lẫn xuống dòng. Văn bản rỗng hoặc chỉ có khoảng trắng trả về `[]` chứ không crash.

**Edge case tôi biết là mình chưa xử lý được:** chữ viết tắt và số thập phân. `"TS. Nguyễn Văn A"` sẽ bị cắt thành hai câu, và `"10.5 kg"` cũng vậy. Với corpus chính sách thương mại điện tử thì ít gặp, nhưng đây là điểm yếu thật của phương pháp tách câu bằng regex thuần.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:

Thuật toán chạy theo thứ tự ưu tiên separator `["\n\n", "\n", ". ", " ", ""]`: cắt bằng ranh giới "to" trước để giữ ngữ nghĩa, chỉ khi mảnh vẫn quá dài mới hạ xuống separator nhỏ hơn.

Điểm tôi đầu tư nhất là **hai chiều**, vì lab doc cảnh báo người ta thường chỉ viết một:

- **Đệ quy xuống:** mảnh nào vẫn dài hơn `chunk_size` thì gọi lại `_split` với danh sách separator còn lại.
- **Gom lên:** các mảnh nhỏ liền kề phải được nối lại cho tới sát `chunk_size`. Thiếu bước này, một file nhiều dòng ngắn sẽ sinh ra hàng trăm chunk vụn.

Có **ba base case**, và tôi viết đủ cả ba: (1) mảnh đã vừa thì trả luôn, (2) hết separator thì cắt cứng theo `chunk_size` bằng `_hard_cut`, (3) separator rỗng hoặc không xuất hiện trong đoạn văn. Test `test_empty_separators_falls_back_gracefully` truyền thẳng `separators=[]` nên thiếu nhánh (3) là fail ngay.

**Một phát hiện khi chạy trên corpus thật:** separator mặc định có `". "`, mà văn bản quy định tiếng Việt đánh số khoản bằng `a.`, `b.`, `2.1.`, `v.`. Khi gặp `"a. Nội dung rất dài..."`, bộ tách cắt ngay tại `". "` và **bỏ rơi nhãn `"a. "` thành mảnh riêng**. Nếu phần nội dung sau đó còn phải đệ quy tiếp, nhãn biến thành chunk 3 ký tự — thuần nhiễu, khớp được gì có nghĩa ở retrieval nhưng vẫn chiếm một slot top-k.

Đo trên `chinh-sach-tra-hang-hoan-tien.md` ở `chunk_size=200`: **48/165 chunk dưới 50 ký tự, nhỏ nhất 3 ký tự**. Tôi thêm helper `_consolidate` để gom mảnh vụn vào hàng xóm nếu tổng vẫn nằm trong `chunk_size`, đưa về **11/130**. Ở `chunk_size=500` và `1000` có đủ khoảng trống nên số stub về **0** — nhóm dùng chunk lớn cho văn bản chính sách thì không gặp vấn đề này.

11 stub còn lại ở `cs=200` là **giới hạn dung lượng, không phải lỗi logic**: tôi kiểm từng ca thì cả hai hàng xóm đều đã đầy 198–200 ký tự, gom vào bên nào cũng vượt ngưỡng.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:

Tôi làm hai helper trước rồi mới làm bốn method công khai, vì làm ngược lại sẽ phải viết lặp cùng một logic bốn lần.

`_make_record` chuẩn hoá một `Document` thành record lưu trong store. Hai chi tiết đáng nói: **copy metadata** thay vì dùng trực tiếp dict của người gọi, và **`metadata.setdefault("doc_id", doc.id)`** — thiếu khoá này thì `delete_document` luôn trả `False` một cách âm thầm.

`_search_records` chạy similarity search trên một tập record bất kỳ. Tôi tách riêng nó ra vì `search()` và `search_with_filter()` **chỉ khác nhau ở tập ứng viên đầu vào**. Cho cả hai đi qua cùng một đường code thì không thể lệch kết quả, và `test_no_filter_returns_all_candidates` đúng là hiển nhiên chứ không phải may mắn. Kết quả trả về tôi bỏ `embedding` đi — vector 384 chiều làm bẩn output khi in ra terminal.

Về ChromaDB: tôi **bỏ hẳn nhánh Chroma**. Code khởi tạo sẵn có một cái bẫy — `self._use_chroma = True` được gán *trước* khi client được tạo, nên nếu máy chấm bài tình cờ có cài `chromadb`, mọi method sẽ rẽ vào nhánh chưa cài đặt và cả 14 test sập.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:

`search_with_filter` **lọc trước rồi mới search**. Đây là điểm tôi thấy quan trọng nhất trong cả lớp: nếu lấy top-k rồi mới bỏ cái không khớp, k slot đã bị chiếm hết bởi tài liệu không hợp lệ và có thể còn lại **0 kết quả** dù store vẫn còn tài liệu hợp lệ.

`delete_document` xoá mọi chunk có `metadata['doc_id']` khớp, trả `True`/`False` tuỳ có xoá được gì không. Vì đã đảm bảo khoá này tồn tại từ `_make_record`, phương thức không bao giờ thất bại im lặng.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:

Ba nhịp: truy xuất top-k → dựng prompt có ngữ cảnh → gọi `llm_fn`.

Phần tôi đầu tư nhất là **cách dựng ngữ cảnh**, vì đây là chỗ quyết định câu trả lời có truy vết được hay không. Mỗi chunk được đánh số `[1] [2] [3]` kèm nguồn (`source_url` → `source` → `doc_id`), và prompt yêu cầu model **trích dẫn số hiệu đoạn đã dùng** khi trả lời. Nhờ vậy câu trả lời truy ngược được về đúng chunk và đúng file — đây là tiêu chí *Source Traceability* trong `docs/EVALUATION.md`, và với corpus quy định thì nó không phải tính năng phụ.

Prompt cũng ràng buộc chống bịa: *"Chỉ sử dụng thông tin trong phần NGỮ CẢNH... không được suy đoán hay thêm kiến thức bên ngoài."*

Và store rỗng thì trả thông báo `NO_CONTEXT_MESSAGE` luôn, **không gọi LLM** — vừa tránh việc model bịa ra câu trả lời khi không có ngữ cảnh, vừa tránh gọi API vô ích.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
platform win32 -- Python 3.11.8, pytest-9.1.1, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.14s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

Ngoài ra `python main.py "Chunking là gì?"` chạy trọn vẹn từ đầu đến cuối.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Tôi dùng `LocalEmbedder` (`paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều) chứ không dùng `MockEmbedder`, vì mock băm MD5 nên điểm số không có ý nghĩa ngữ nghĩa.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người mua có thể gửi yêu cầu trả hàng trong vòng **15 ngày** kể từ khi nhận hàng. | Thời hạn để gửi yêu cầu hoàn tiền là **mười lăm ngày** sau khi nhận được sản phẩm. | cao | **+0.8428** | ✅ |
| 2 | Shopee hiện chưa hỗ trợ yêu cầu đổi hàng. | Nền tảng chỉ tiếp nhận trả hàng và hoàn tiền, không đổi sang sản phẩm khác. | cao | **+0.3644** | ❌ (thực tế trung bình) |
| 3 | **Người bán** sẽ chịu chi phí vận chuyển chiều hoàn trả sản phẩm. | **Người mua** không phải thanh toán bất cứ chi phí vận chuyển nào cho việc trả hàng. | trung bình | **+0.4056** | ✅ |
| 4 | Hàng cồng kềnh là đơn hàng có khối lượng từ 10kg trở lên. | Hôm nay trời Hà Nội mát mẻ và có mưa nhẹ vào buổi chiều. | thấp | **−0.0162** | ✅ |
| 5 | Mã giảm giá sẽ được hoàn lại trong vòng 48 giờ. | Thẻ tín dụng và thẻ ghi nợ là phương thức thanh toán phổ biến. | thấp | **+0.0469** | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

Bất ngờ nhất là **cặp 2**. Tôi chọn nó đúng theo gợi ý trong lab doc — *"hai câu khác từ vựng nhưng cùng nghĩa"* — và kỳ vọng nó sẽ đạt điểm cao, chứng minh mô hình nắm được nghĩa. Thực tế nó chỉ đạt **0.3644**, **thấp hơn cả cặp 3** (0.4056) — trong khi cặp 3 là hai câu nói về **hai đối tượng trái ngược nhau** (người bán chịu phí so với người mua không phải trả phí).

Điều này nói lên một giới hạn thật của embeddings mà tôi chưa ý thức rõ trước khi làm bài: **vector nhúng không biểu diễn ngữ nghĩa thuần tuý — nó vẫn phụ thuộc đáng kể vào trùng lặp từ vựng.** Cặp 3 đạt điểm cao hơn không phải vì hai câu cùng nghĩa (chúng thậm chí mâu thuẫn về chủ thể), mà vì chúng chia sẻ một cụm từ dài gần như nguyên vẹn: *"chi phí vận chuyển … hoàn trả sản phẩm"*. Ngược lại, cặp 2 tuy cùng nghĩa nhưng gần như không chia sẻ từ vựng nào.

Hệ quả trực tiếp cho phần truy xuất: **cosine đo độ giống chủ đề, không đo mật độ thông tin trả lời được.** Một chunk đúng chủ đề nhưng rỗng số liệu vẫn có thể thắng chunk chứa đáp án — và đúng như vậy, đây chính là cơ chế gây ra lỗi ở câu hỏi Q2 mà tôi phân tích trong `REPORT_NHOM.md`.

Cặp 1 cũng đáng chú ý theo hướng ngược lại: hai câu viết con số khác nhau hoàn toàn (`15` so với `mười lăm`) vẫn đạt 0.84, cho thấy mô hình **có** hiểu nghĩa ở mức độ nào đó. Vậy nên kết luận không phải "embedding vô dụng" mà là: nó hiểu nghĩa **khi có đủ điểm tựa từ vựng**, và yếu đi rõ rệt khi hai câu cùng nghĩa nhưng khác từ hoàn toàn.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân, chiến lược `HeadingChunker(max_section_size=800)`, embedder `paraphrase-multilingual-MiniLM-L12-v2` (384 chiều). Corpus: 15 tài liệu → 561 chunk.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan? | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Đơn giao thực phẩm tươi sống và đông lạnh phải gửi yêu cầu trả hàng trong bao lâu? | `quy-dinh-chung-tra-hang-hoan-tien` mục *1.2. Thời gian tối đa để gửi yêu cầu…* — chứa trọn con số **24 giờ** | **+0.770** | ✅ hạng 1 | *"Trong vòng 24 giờ kể từ khi đơn hàng được cập nhật trạng thái 'Giao hàng thành công' [1]"* |
| 2 | Tính năng Trả hàng COM (đổi ý) áp dụng cho nhóm người mua nào, từ thời điểm nào? | `chinh-sach-tra-hang-hoan-tien` mục *4. QUY ĐỊNH BỔ SUNG ĐỐI VỚI CÁC TRƯỜNG HỢP TRẢ HÀNG COM* — **đúng chủ đề nhưng KHÔNG chứa 24/11/2025** | +0.784 | ️ đúng chủ đề, thiếu đáp án | *"Không tìm thấy thông tin về nhóm người mua và thời điểm áp dụng trong ngữ cảnh được cung cấp"* |
| 3 | Phí vận chuyển chiều hoàn trả do bên nào chịu khi Shopee chấp thuận? | `chinh-sach-tra-hang-hoan-tien-nguoi-ban` mục *7. Trách nhiệm về chi phí vận chuyển hoàn trả sản phẩm của Người bán* | **+0.836** | ✅ hạng 1 | *"Người Bán chịu chi phí khi đơn được Shopee chấp thuận mà không do lỗi của Người Mua hoặc đơn vị vận chuyển [1]"* |
| 4 | Mã giảm giá được hoàn lại trong vòng bao lâu? | `quy-dinh-chung-tra-hang-hoan-tien` mục *2. Quy định chung về việc hoàn lại Mã giảm giá/Shopee Xu* — chứa **48 giờ** | **+0.750** | ✅ hạng 1 | *"Trong vòng 48 giờ, không kể Thứ Bảy, Chủ Nhật và ngày lễ, kể từ khi yêu cầu được chấp nhận hoàn tiền [1]"* |
| 5 | Đóng gói hàng hoàn trả cần chuẩn bị gì và lưu ý gì? | Top-1 là `chinh-sach-tra-hang-hoan-tien` mục *6. YÊU CẦU ĐỐI VỚI SẢN PHẨM HOÀN TRẢ* (đúng chủ đề, **thiếu chi tiết đóng gói**). Chunk chứa đáp án là `cach-dong-goi-don-hoan-tra` — **hộp vận chuyển bên ngoài** | +0.709 top-1<br>(+0.686 ở hạng 2) | ✅ (hạng 2) | *"Cần vật liệu đóng gói, phiếu gửi hàng, quay video quá trình đóng gói; dùng thêm hộp vận chuyển bên ngoài và không dán lên hộp nhà sản xuất [2]"* |

> **Ghi chú về cột "Câu trả lời của Agent":** các câu trả lời trên là **kết quả kỳ vọng** suy ra từ ngữ cảnh truy xuất được, dựa đúng trên prompt mà `KnowledgeBaseAgent` dựng (đánh số `[n]`, ràng buộc chỉ dùng ngữ cảnh, bắt buộc trích dẫn nguồn). `main.py` hiện dùng `demo_llm` là hàm giả nên chưa sinh được câu trả lời thật — **cần cắm một LLM thật để chốt cột này**. Phần đánh giá điểm số bên trên đã tính theo `evidence_rank` trên ngữ cảnh truy xuất, không phụ thuộc LLM.

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **4 / 5**

**Chỉ số tổng hợp** (theo cách chấm của `docs/SCORING.md`, 2 điểm/câu):

```
Hit@1 = 3/5 (60%)   ·   Hit@3 = 4/5 (80%)   ·   MRR = 0.700   ·   lab_score = 7/10
```

**Một ghi chú về độ nhạy của phép đo:** câu Q5 ban đầu bị thiếu một ký tự có dấu. Sau khi sửa cho đúng chính tả, hạng của chunk chứa đáp án ở câu đó **tụt từ 1 xuống 2**, kéo `lab_score` từ 8/10 xuống 7/10. Một ký tự trong đề bài đổi kết quả của cả một câu — với bộ 5 câu, mỗi câu chiếm 20% số điểm nên độ nhạy này là đáng kể, nhóm cần ghi nhớ khi diễn giải số liệu.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

*(bổ sung sau khi demo)*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 |
| **Tổng phần cá nhân** | **56 / 60** |