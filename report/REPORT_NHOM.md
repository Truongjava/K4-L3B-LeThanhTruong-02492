# Bao Cao Nhom - Lab 7: Embedding & Vector Store

**Nhom:** G08
**Thanh vien:** Nguyễn Đình Anh Đức, Lê Thanh Trường, Hoàng Văn Dương
**Ngay:** 20/09/2026

> Nop 1 ban / nhom. Phan ca nhan moi thanh vien nop rieng trong `REPORT_CANHAN.md`. Chi tiet thang diem: `docs/SCORING.md`.

**Tong diem phan nhom: 40** = Lua chon tai lieu (10) + Thiet ke chien luoc (15) + Chat luong truy xuat (10) + Thuyet trinh (5).

---

## 1. Lua chon tai lieu (Document Set Quality) - Nhom (10 diem)

### Chu de (Domain) & Ly do chon

**Chu de:** Chinh sach Tra hang / Hoan tien cua Shopee Viet Nam.

Nhom chon chu de nay vi day la mot mien du lieu co tinh thuc te cao, nhieu cau hoi nguoi dung can cau tra loi chinh xac theo dieu kien, thoi han, phi van chuyen va bang chung. Tai lieu cung phu hop de kiem thu RAG vi co nhieu doan chinh sach dai, bang thoi gian hoan tien, nhieu doi tuong ap dung nhu `buyer`, `seller`, `both`, va nhieu thong tin de bi mat neu chunking khong tot.

### Danh sach tai lieu (Data Inventory)

Corpus chinh cua nhom nam trong `data/chinh-sach-doi-tra`, gom 14 file Markdown duoc crawl tu Shopee Help Center. Cac tai lieu deu co frontmatter metadata gom `doc_id`, `title`, `source_url`, `retrieved_at`, `document_version`, `audience`, `category`, `language`, `license_or_permission`.

| # | Ten tai lieu | Nguon (Source URL) | Ngay lay / Phien ban | So ky tu | Metadata da gan |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Shopee article 188931 | https://help.shopee.vn/portal/4/article/188931 | 2026-09-20 / not-stated | 6,298 | buyer, return-refund-policy, vi |
| 2 | Shopee article 189473 | https://help.shopee.vn/portal/4/article/189473 | 2026-09-20 / not-stated | 3,869 | buyer, return-refund-policy, vi |
| 3 | Shopee article 189477 | https://help.shopee.vn/portal/4/article/189477 | 2026-09-20 / not-stated | 5,903 | buyer, return-shipping-fee, vi |
| 4 | Shopee article 190242 | https://help.shopee.vn/portal/4/article/190242 | 2026-09-20 / not-stated | 8,084 | buyer, return-refund-policy, vi |
| 5 | Shopee article 195504 | https://help.shopee.vn/portal/4/article/195504 | 2026-09-20 / not-stated | 18,726 | buyer, return-refund-policy, vi |
| 6 | Shopee article 77243 | https://help.shopee.vn/portal/4/article/77243 | 2026-09-20 / not-stated | 83,370 | both, terms-of-service, vi |
| 7 | Shopee article 77245 | https://help.shopee.vn/portal/4/article/77245 | 2026-09-20 / not-stated | 77,839 | buyer, return-refund-policy, vi |
| 8 | Shopee article 77251 | https://help.shopee.vn/portal/4/article/77251 | 2026-09-20 / not-stated | 19,598 | buyer, return-refund-policy, vi |
| 9 | Shopee article 77262 | https://help.shopee.vn/portal/4/article/77262 | 2026-09-20 / not-stated | 33,723 | buyer, return-refund-policy, vi |
| 10 | Shopee article 77265 | https://help.shopee.vn/portal/4/article/77265 | 2026-09-20 / not-stated | 4,812 | buyer, return-refund-policy, vi |
| 11 | Shopee article 77484 | https://help.shopee.vn/portal/4/article/77484 | 2026-09-20 / not-stated | 22,984 | buyer, return-refund-policy, vi |
| 12 | Shopee article 79233 | https://help.shopee.vn/portal/4/article/79233 | 2026-09-20 / not-stated | 2,499 | buyer, return-request-guide, vi |
| 13 | Shopee article 79467 | https://help.shopee.vn/portal/4/article/79467 | 2026-09-20 / not-stated | 3,430 | buyer, return-refund-policy, vi |
| 14 | Shopee article 79508 | https://help.shopee.vn/portal/4/article/79508 | 2026-09-20 / not-stated | 3,608 | buyer, return-refund-policy, vi |

**Danh sach kiem tra quan tri du lieu (Data governance checklist):**
- [x] Corpus chi chua nguon cong khai tu Shopee Help Center, khong chua du lieu ca nhan, thong tin dang nhap hoac tai lieu noi bo.
- [x] Moi tai lieu co `source_url`, `retrieved_at`, `document_version` trong metadata.
- [x] Du lieu da duoc luu thanh Markdown kem frontmatter de de trace nguon va loc metadata.
- [x] Noi dung crawl con co nhieu doan nhieu tu trang web, vi vay can lam sach va chon chunking phu hop voi van ban chinh sach.

### Cau truc Metadata (Metadata Schema)

| Truong metadata | Kieu | Vi du gia tri | Tai sao huu ich cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `shopee-article-77251` | Dinh danh tai lieu, dung de trace nguon, xoa document va doi chieu gold doc. |
| `title` | string | `Shopee article 77251` | Giu ngu canh khi chunk ngan hoac khi nhieu chinh sach co noi dung gan nhau. |
| `source_url` | string | `https://help.shopee.vn/portal/4/article/77251` | Cho phep kiem chung nguon cong khai sau khi agent tra loi. |
| `retrieved_at` | date string | `2026-09-20` | Biet thoi diem crawl, quan trong vi chinh sach thuong mai dien tu co the thay doi. |
| `document_version` | string | `not-stated` | Ghi nhan phien ban/ngay hieu luc neu nguon co cong bo. |
| `audience` | string | `buyer`, `seller`, `both` | Loc theo doi tuong de tranh lay nham chinh sach cua nguoi mua/nguoi ban. |
| `category` | string | `return-refund-policy` | Loc theo nhom nghiep vu nhu phi hoan tra, huong dan gui yeu cau, dieu khoan dich vu. |
| `language` | string | `vi` | Ho tro chon embedder/tokenizer phu hop voi tieng Viet. |
| `license_or_permission` | string | `public-source` | Ghi nhan quyen su dung nguon du lieu cong khai cho bai lab. |

---

## 2. Thiet ke chien luoc (Strategy Design) - Nhom (15 diem)

### Phan tich duong co so (Baseline Analysis)

Nhom dung 5 cau hoi danh gia thong nhat va tinh diem theo `docs/SCORING.md`: moi cau toi da 2 diem, top-3 co chunk chua evidence thi duoc tinh hit. Benchmark hien tai dung `LexicalHashEmbedder` de chay offline on dinh; benchmark cua Truong dung `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

| Chien luoc | Corpus / Embedder | So chunk | Do dai TB | Hit@1 | Hit@3 | MRR | Diem |
|-----------|-------------------|----------|-----------|-------|-------|-----|------|
| `fixed_size` | Duc, lexical-hash | 90 | 481.2 | 60% | 60% | 0.600 | 6/10 |
| `by_sentences` | Duc, lexical-hash | 71 | 549.1 | 80% | 100% | 0.900 | 9/10 |
| `recursive` | Duc, lexical-hash | 96 | 405.4 | 60% | 60% | 0.600 | 6/10 |
| `HeadingChunker` | Truong, multilingual MiniLM | 561 | chua ghi | 60% | 80% | 0.700 | 7/10 |
| `fixed_size` | Corpus hien tai, lexical-hash | 659 | 496.2 | 20% | 60% | 0.400 | 4/10 |
| `by_sentences` | Corpus hien tai, lexical-hash | 630 | 464.5 | 80% | 80% | 0.800 | 8/10 |
| `recursive` | Corpus hien tai, lexical-hash | 767 | 380.8 | 40% | 40% | 0.400 | 4/10 |
| `policy_structure` | Corpus hien tai, lexical-hash | 1,273 | 284.2 | 80% | 100% | 0.867 | 9/10 |

Nhan xet chinh: chunk theo cau va chunk theo cau truc chinh sach deu tot hon fixed-size/recursive mac dinh. Fixed-size de cat ngang mot dieu kien hoac dong bang; recursive mac dinh giu gioi han do dai tot nhung co the tao cac doan khong du ngu canh khi van ban crawl co nhieu dong roi. Voi du lieu chinh sach Shopee, chunk can du ngan de khong lan nhieu dieu kien, nhung van phai giu tieu de tai lieu de biet doan do thuoc chinh sach nao.

### Chien luoc cua tung thanh vien

**Thanh vien 1 - Le Thanh Truong**
- **Loai chien luoc:** custom `HeadingChunker(max_section_size=800)`.
- **Mo ta & ly do chon:** Chien luoc uu tien giu cac muc chinh sach theo heading/muc so, vi van ban Shopee thuong co cau truc dieu khoan nhu `1.2`, `7.1`, `8.1`. Ket qua dat `Hit@1 = 60%`, `Hit@3 = 80%`, `MRR = 0.700`, `lab_score = 7/10` voi embedding multilingual MiniLM. Diem yeu la Q2 khong tim duoc moc `24/11/2025`, va Q5 chi dua evidence len hang 2.

**Thanh vien 2 - Nguyen Dinh Anh Duc**
- **Loai chien luoc:** so sanh `fixed_size`, `by_sentences`, `recursive`.
- **Mo ta & ly do chon:** Duc benchmark cac chien luoc mac dinh bang lexical-hash offline. Ket qua tot nhat la `by_sentences` voi `71 chunks`, `avg_len = 549.1`, `Hit@1 = 80%`, `Hit@3 = 100%`, `MRR = 0.900`, `lab_score = 9/10`. Dieu nay cho thay voi chinh sach dang FAQ, don vi cau thuong chua du y tra loi hon la cat theo ky tu.

**Thanh vien 3 - [bo sung ten]**
- **Loai chien luoc:** custom `PolicyStructureChunker(chunk_size=450, max_sentences=2)`.
- **Mo ta & ly do chon:** Chien luoc nay la hybrid: tach theo cau de giu don vi ngu nghia, dung recursive fallback khi cau/bang qua dai, gioi han moi chunk toi da 450 ky tu, va lap lai tieu de tai lieu trong tung chunk. Tren corpus hien tai, chien luoc dat `Hit@1 = 80%`, `Hit@3 = 100%`, `MRR = 0.867`, `lab_score = 9/10`. Doi lai so chunk tang len 1,273 nen chi phi embedding/luu tru cao hon.

### So sanh giua cac thanh vien

| Thanh vien | Chien luoc (Strategy) | Diem truy xuat (/10) | Diem manh | Diem yeu |
|-----------|----------|----------------------|-----------|----------|
| Le Thanh Truong | `HeadingChunker` + multilingual MiniLM | 7/10 | Giu duoc cau truc muc/dieu khoan, co semantic embedding that. | Miss Q2, Q5 khong o top-1; phu thuoc heading sach. |
| Nguyen Dinh Anh Duc | `by_sentences` | 9/10 | Gon, it chunk, giu cau day du, Hit@3 100% tren benchmark cua Duc. | Co the tao chunk rat dai neu van ban/bang it dau cau. |
| [bo sung ten] | `PolicyStructureChunker` | 9/10 | Hit@3 100%, xu ly tot cau ngan + bang/doan dai, chunk co tieu de nguon. | Nhieu chunk hon, tang chi phi index va embedding. |

**Chien luoc nao tot nhat cho chu de nay? Tai sao?**

Voi du lieu Shopee hien tai, nhom chon `PolicyStructureChunker(chunk_size=450, max_sentences=2)` lam cau hinh khuyen nghi. Ly do la no dat `9/10`, dua ca 5 cau hoi co evidence vao top-3, dong thoi tranh nhuoc diem cua `by_sentences` la co chunk dai toi 3,659 ky tu khi gap bang hoac doan crawl thieu dau cau. Chien luoc nay phu hop voi chinh sach thuong mai dien tu vi moi cau/dieu kien thuong ngan, nhieu thong tin quan trong la so lieu nhu `24 gio`, `7 - 14 ngay`, `3 - 5 ngay`, `48 gio`.

---

## 3. Cau hoi danh gia & Chat luong truy xuat (Retrieval Quality) - Nhom (10 diem)

### Cau hoi danh gia & Cau tra loi chuan (nhom thong nhat)

| # | Cau hoi (Query) | Cau tra loi chuan (Gold Answer) | Chunk nao chua thong tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Thuc pham tuoi song va dong lanh phai gui yeu cau tra hang/hoan tien trong bao lau? | Trong vong 24 gio ke tu khi don hang duoc cap nhat giao hang thanh cong. | `shopee-article-77251`, marker `24 gio` |
| 2 | Sau khi Shopee chap nhan hoan tien, the tin dung hoac ghi no nhan tien trong bao lau? | Tu 7 den 14 ngay lam viec, tuy theo ngan hang. | `shopee-article-189473`, marker `7 - 14 ngay lam viec` |
| 3 | Neu tu sap xep gui hang hoan tra thi nguoi mua co phai tra phi truoc khong? | Co. Nguoi mua tra phi truoc; Shopee ho tro hoan phi trong 3-5 ngay lam viec neu du dieu kien. | `shopee-article-77251`, marker `Tu sap xep` / `thanh toan truoc` |
| 4 | Nguoi mua chua nhan duoc hang thi can cung cap bang chung gi? | Khong can cung cap bang chung; Shopee xu ly dua tren he thong theo doi don hang. | `shopee-article-79467`, marker `khong can cung cap bat ky bang chung nao` |
| 5 | Khi dong goi hang hoan tra, nguoi mua can quay video va gui kem nhung gi? | Quay video dong goi va gui du hop, giay to, phu kien, qua tang di kem neu co. | `shopee-article-79508`, marker `quay video qua trinh dong goi` |

### Tong hop chat luong truy xuat cua nhom

Ket qua duoi day lay theo chien luoc tot nhat tren corpus hien tai: `PolicyStructureChunker`.

| # | Cau hoi | Chien luoc tot nhat cho cau nay | Co chunk lien quan trong top-3? | Ghi chu |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Thuc pham tuoi song/dong lanh gui yeu cau trong bao lau? | `policy_structure` | Co, hang 1 | Chunk chua ro `24 gio`. |
| 2 | The tin dung/ghi no nhan tien hoan trong bao lau? | `policy_structure` | Co, hang 3 | Co evidence nhung chua len top-1 vi chunk khac cung tai lieu noi chung ve hoan tien. |
| 3 | Tu sap xep gui hang hoan tra co phai tra phi truoc khong? | `policy_structure` | Co, hang 1 | Chunk chua dieu kien thanh toan truoc va hoan phi. |
| 4 | Chua nhan duoc hang can cung cap bang chung gi? | `policy_structure` | Co, hang 1 | Cau hoi khop truc tiep voi noi dung evidence. |
| 5 | Dong goi hang hoan tra can quay video/gui kem gi? | `policy_structure` | Co, hang 1 | Chien luoc lap title giup chunk ngan van giu dung ngu canh dong goi. |

**Loc bang metadata co giup ich khong? O cau hoi nao?**

Metadata filter huu ich ve mat thiet ke vi giup gioi han khong gian tim kiem theo `audience` hoac `category` truoc khi tinh similarity. Tuy nhien trong corpus hien tai, phan lon tai lieu co `audience=buyer`, chi co mot so it `both`, nen filter `audience=buyer` khong lam thay doi top-3 o Q1. Voi bo du lieu can bang hon giua `buyer` va `seller`, filter se quan trong hon, dac biet o cac cau hoi de nham giua chi phi cua Nguoi Mua va trach nhiem cua Nguoi Ban.

---

## 4. Thuyet trinh (Demo) & Bai hoc nhom - Nhom (5 diem)

**Nhung phan tich (insights) hay nhat nhom se trinh bay:**
- Mock embedding chi dung de kiem thu pipeline, khong phan anh chat luong retrieval that; benchmark can it nhat dung lexical-hash hoac embedding model that.
- Chunking theo cau tot hon fixed-size vi chinh sach Shopee chua nhieu cau ngan co so lieu quan trong, nhung can fallback khi gap bang/doan crawl dai.
- `PolicyStructureChunker` dat ket qua tot vi moi chunk vua ngan vua co title nguon, giup truy xuat dung ca cau hoi so lieu, dieu kien va huong dan dong goi.

**Bai hoc rut ra khi so sanh trong nhom:**

Cung mot bo tai lieu nhung chien luoc chunking lam thay doi manh chat luong truy xuat: fixed-size tren corpus hien tai chi dat 4/10, trong khi `policy_structure` dat 9/10. Nhom cung thay rang diem tot khong chi phu thuoc so chunk; quan trong hon la chunk co giu dung don vi tra loi khong va co du ngu canh nguon khong.

**Neu lam lai, nhom se thay doi gi trong chien luoc du lieu (data strategy)?**

Nhom se lam sach du lieu crawl ky hon, dac biet la menu/footer va cac bang bi chuyen thanh dong text roi. Ngoai ra nen gan metadata chi tiet hon, vi du `audience=seller` cho tai lieu nguoi ban, `category=refund-timing`, `category=packing-guide`, va them chunking rieng cho bang: moi chunk giu header bang + mot nhom dong de khong mat quan he giua cot va gia tri.

---

## Tu danh gia (Phan Nhom)

| Tieu chi | Diem tu danh gia |
|----------|-------------------|
| Lua chon tai lieu (Document Set Quality) | 9 / 10 |
| Thiet ke chien luoc (Strategy Design) | 14 / 15 |
| Chat luong truy xuat (Retrieval Quality) | 9 / 10 |
| Thuyet trinh (Demo) | 4 / 5 |
| **Tong phan nhom** | **36 / 40** |
