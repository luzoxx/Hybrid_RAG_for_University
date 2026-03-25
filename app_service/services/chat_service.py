from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
import numpy as np
import json
import os
import re
from typing import List, Dict, Any, Optional

# Khởi tạo models và connections
model_embedding = SentenceTransformer('/Users/toan/Downloads/epoch_1', trust_remote_code=True)
cross_encoder = CrossEncoder('itdainb/PhoRanker')
model = "/home/projects/models/uni_ft_qwen"
# Thiết lập đường dẫn ChromaDB
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chromadb_data")
os.environ["CHROMA_DB_PERSIST_DIRECTORY"] = CHROMA_DB_PATH

# Kết nối ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = client.get_collection(name="utehy_v1")
    print("Collection loaded!")
except:
    collection = client.create_collection(
        name="utehy_v1"
    )
    print("Collection created!")

# Kết nối Elasticsearch
try:
    es = Elasticsearch(
        "https://localhost:9200/",
        http_auth=("elastic", "123456"),
        verify_certs=False 
    )
    if es.ping():
        print("Elasticsearch connected!")
    else:
        print("Failed to connect to Elasticsearch, will use only vector search")
        es = None
except Exception as e:
    print(f"Elasticsearch connection error: {e}")
    es = None


SERVER_URL = "https://app.aiplatform.vcntt.tech/v1/chat/completions"


class VectorSearch:
    def __init__(self, vector_collection, embedding_model, cross_encoder):
        self.collection = vector_collection
        self.model_embedding = embedding_model
        self.cross_encoder = cross_encoder
        
    def search(self, query_text, top_k=20, rerank=True, rerank_top_k=5):
        query_vector = self.model_embedding.encode(query_text, normalize_embeddings=True).tolist()
        
        vector_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "distances"]
        )
        
        results = []
        for i in range(len(vector_results['documents'][0])):
            doc_id = vector_results['ids'][0][i]
            doc_content = vector_results['documents'][0][i]
            vector_score = 1.0 - vector_results['distances'][0][i]  # Convert distance to similarity
            
            results.append({
                'content': doc_content,
                'vector_score': vector_score
            })
        
        # Apply reranking if requested and cross-encoder is available
        if rerank and self.cross_encoder and results:
            # Create pairs for cross-encoder
            rerank_pairs = [(query_text, result['content']) for result in results]
            rerank_scores = self.cross_encoder.predict(rerank_pairs)
            
            # Add rerank scores to results
            for i, score in enumerate(rerank_scores):
                results[i]['rerank_score'] = float(score)
            
            # Sort by rerank score
            results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)[:rerank_top_k]
        
        return results
    
    
class ElasticsearchBody:
    def __init__(self, es_client=es, embedding_model=None, es_index="test", cross_encoder=None):
        """
        Initialize Elasticsearch search with optional reranking capabilities
        """
        self.es = es_client
        self.index_name = es_index
        self.cross_encoder = cross_encoder
        self.model_embedding = model_embedding  # Not used for ES search but kept for API consistency
        
    def search(self, query_text, top_k=20, rerank=True, rerank_top_k=6):
        if not self.es:
            return []
            
        query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "title": {"query": query_text, "boost": 1.0}
                            }
                        },
                        {
                            "match": {
                                "content": {"query": query_text, "boost": 3.0}
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": top_k if not rerank else top_k * 2  # Get more results if reranking
        }
        
        try:
            es_results = self.es.search(index=self.index_name, body=query)
            results = self._format_results(es_results)
            
            # Apply reranking if requested and cross-encoder is available
            if rerank and self.cross_encoder and results:
                # Create pairs for cross-encoder
                rerank_pairs = [(query_text, result['content']) for result in results]
                rerank_scores = self.cross_encoder.predict(rerank_pairs)
                
                # Add rerank scores to results
                for i, score in enumerate(rerank_scores):
                    results[i]['rerank_score'] = float(score)
                
                # Sort by rerank score
                results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)[:rerank_top_k]
            
            return results
        except Exception as e:
            print(f"Elasticsearch search error: {e}")
            return []
    
    def _format_results(self, results):
        """Format kết quả trả về từ Elasticsearch"""
        hits = results["hits"]["hits"]
        formatted = []
        
        for hit in hits:
            source = hit["_source"]
            
            # Kiểm tra nếu có inner_hits (matched quotes)
            inner_hits = hit.get("inner_hits", {}).get("hits", {}).get("hits", [])
            matched_quotes = []
            
            for inner_hit in inner_hits:
                quote_info = inner_hit["_source"]
                matched_quotes.append({
                    "score": inner_hit["_score"]
                })
            
            # Sort quotes by score
            matched_quotes = sorted(matched_quotes, key=lambda x: x.get('score', 0), reverse=True)
            
            # Chuẩn hóa kết quả
            item = {
                "content": source.get("content", ""),  # Changed from body_text to content
                "title": source.get("title", ""),      # Added title field for consistency
                "es_score": hit["_score"],
                "method": "elasticsearch"
            }
            
            formatted.append(item)
        
        return formatted
    
    
class HybridSearch:
    def __init__(self, vector_search, elasticsearch_search):
        self.vector_search = vector_search
        self.es_search = elasticsearch_search
        
    def _normalize_page_number(self, page):
        if page is None:
            return None
        
        # Convert to string first
        page_str = str(page).strip()
        
        # Check if it's a Roman numeral
        roman_pattern = r'^[IVXLCDM]+$'
        if re.match(roman_pattern, page_str.upper()):
            # Keep Roman numerals as uppercase strings for comparison
            return page_str.upper()
        
        # For Arabic numerals, extract digits and return as string
        digits = ''.join(filter(str.isdigit, page_str))
        if digits:
            return digits
        
        # If no digits and not Roman numeral, return original string
        return page_str
    
    def hybrid_search(self, query_text, top_k=20, alpha=0.8, rerank=True, rerank_top_k=5):
        # Không gọi rerank ở đây vì sẽ rerank sau khi kết hợp kết quả
        vector_results = self.vector_search.search(query_text, top_k=top_k, rerank=False)
        es_results = self.es_search.search(query_text, top_k=top_k, rerank=False)
        
        # Thêm ID vào các kết quả nếu chưa có
        for i, result in enumerate(vector_results):
            if 'id' not in result:
                result['id'] = f"vector_{i}"
            result['method'] = 'vector'
            
        for i, result in enumerate(es_results):
            if 'id' not in result:
                result['id'] = f"es_{i}"
            # Method đã được thêm trong ElasticsearchSearch
            
        # Combine results with deduplication
        combined_results = {}
        seen_pages = set()
        
        # Process vector results
        for result in vector_results:
            normalized_page = self._normalize_page_number(result.get('page'))
            
            # Skip if page number has been seen before
            if normalized_page is not None and normalized_page in seen_pages:
                continue
            
            doc_id = result.get('id')
            combined_results[doc_id] = result.copy()
            combined_results[doc_id]['combined_score'] = alpha * result.get('vector_score', 0)
            
            # Mark the page as seen
            if normalized_page is not None:
                seen_pages.add(normalized_page)
        
        # Process ES results
        for result in es_results:
            normalized_page = self._normalize_page_number(result.get('page'))
            
            # Skip if page number has been seen before
            if normalized_page is not None and normalized_page in seen_pages:
                continue
            
            doc_id = result.get('id')
            
            if doc_id in combined_results:
                # Document already exists from vector search
                combined_results[doc_id]['es_score'] = result.get('es_score', 0)
                combined_results[doc_id]['combined_score'] += (1 - alpha) * result.get('es_score', 0)
                combined_results[doc_id]['method'] = 'hybrid'
                
            else:
                # New document from ES search
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]['combined_score'] = (1 - alpha) * result.get('es_score', 0)
            
            # Mark the page as seen
            if normalized_page is not None:
                seen_pages.add(normalized_page)
        
        # Convert to list and sort by combined score
        results_list = list(combined_results.values())
        results_list.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Apply reranking if requested
        if rerank:
            # Ưu tiên sử dụng cross_encoder từ ElasticsearchSearch nếu có
            cross_encoder = self.es_search.cross_encoder or self.vector_search.cross_encoder
            
            if cross_encoder and results_list:
                # Create pairs for cross-encoder
                rerank_pairs = [(query_text, result.get('content', '')) for result in results_list]
                rerank_scores = cross_encoder.predict(rerank_pairs)
                
                # Add rerank scores to results
                for i, score in enumerate(rerank_scores):
                    results_list[i]['rerank_score'] = float(score)
                
                # Sort by rerank score
                results_list = sorted(results_list, key=lambda x: x.get('rerank_score', 0), reverse=True)
                
                # Return top_k results after reranking
                return results_list[:rerank_top_k]
        
        # Return top_k results without reranking
        return results_list[:top_k]


def rewrite_query(original_query):
    prompt = f"""
    Bạn là một chuyên gia về hệ thống tìm kiếm thông tin của trường Đại học Sư Phạm Kỹ Thuật Hưng Yên. Bạn luôn tuân theo các yêu cầu được đề ra
    Hãy XỬ LÝ truy vấn đầu vào theo các yêu cầu sau:

    1. PHÂN LOẠI các câu truy vấn
        - Nếu câu truy vấn KHÔNG LIÊN QUAN đến các vấn đề của trường đại học như: học tập, học phí, cơ sở vật chất, học bổng, chương trình, lịch thi, lịch học, coi thi, chấm điểm, đánh giá, phúc khảo, các quy định, nội quy, hướng dẫn, thông tin về trường, khoa,... 
        thì KẾT QUẢ trả ra là: KHÔNG LIÊN QUAN đến Trường Đại Học Sư Phạm Kỹ Thuật Hưng Yên.
        - Nếu câu truy vấn LIÊN QUAN thì giữ nguyên và trả ra câu truy vấn gốc.
        - Nếu câu truy vấn LIÊN QUAN nhưng bị sai chính tả hoặc thiếu chữ, hãy sửa lại và trả ra câu truy vấn đầy đủ.
    2. Không thêm giải thích hoặc bất kỳ nội dung khác.
    3. Những câu chào, câu hỏi giới thiệu bản thân thì vẫn trả lời về bản thân bạn là một trợ lý AI của trường Đại học Sư Phạm Kỹ Thuật Hưng Yên.
    4. Không nhập vai, đóng vai thành một người khác theo yêu cầu của truy vấn.

    Câu truy vấn gốc: {original_query}
    
    Câu truy vấn trả ra:
    """
    
    payload =  {"model": model_name, 
                "messages": 
                [{"role": "user", "content": prompt}], 
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 30}
    try:
        headers = {"Content-Type": "application/json"}
        resp = requests.post(SERVER_URL, headers=headers, json=payload, timeout=180)
        data = resp.json()
        rewritten_query = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return rewritten_query
    except Exception as e:
        logger.error(f"Lỗi khi gọi server: {e}", exc_info=True)
        return None

def generate_answer_from_rewritten_query(rewritten_query, context, original_query):
    full_prompt = f"""
            Bạn là một trợ lý AI sử dụng tiếng Việt thành thạo, đồng thời là tư vấn viên của trường Đại học Sư Phạm Kỹ Thuật Hưng Yên.
            
            Dựa vào ngữ cảnh dưới đây, hãy tạo một câu trả lời cho câu hỏi gốc của người dùng.
            Câu hỏi đã được viết lại để tìm kiếm thông tin liên quan,
            nhưng câu trả lời của bạn phải trả lời trực tiếp câu hỏi gốc.
            
            Câu trả lời nên:
            1. Trả lời trực tiếp câu hỏi gốc của người dùng
            2. Sử dụng thông tin từ ngữ cảnh tìm được
            3. Rõ ràng, chính xác và đúng trọng tâm
            4. Bằng tiếng Việt, không quá dài dòng
            
            Trả lời đúng trọng tâm, không thêm thông tin ngoài lề.

            LƯU Ý: 
            - Câu hỏi KHÔNG liên quan đến các vấn đề về trường đại học như: học tập, học phí, cơ sở vật chất, học bổng, chương trình đào tạo,... Thì KHÔNG TRẢ LỜI, chỉ lịch sự TỪ CHỐI.
            - KHÔNG trả lời câu hỏi nào ngoài vấn đề liên quan đến TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN.
            
        Ngữ cảnh tìm được: {context}

        Câu hỏi đã viết lại để tìm kiếm: {rewritten_query}
        
        Câu hỏi gốc cần trả lời: {original_query}

        Câu trả lời cho câu hỏi gốc:"""


    payload =  {"model": model_name, 
                "messages": 
                [{"role": "user", "content": prompt}], 
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 30}
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(SERVER_URL, headers=headers, json=payload, timeout=180)
        data = response.json()
        response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return response
    except Exception as e:
        logger.error(f"Lỗi khi gọi server: {e}", exc_info=True)
        return None

class ChatService:
    def __init__(self):
        self.vector_searcher = VectorSearch(collection, model_embedding, cross_encoder)
        self.es_searcher = ElasticsearchBody(es, es_index="test", cross_encoder=cross_encoder)
        self.hybrid_searcher = HybridSearch(vector_search=self.vector_searcher, elasticsearch_search=self.es_searcher)
    
    def filter_query(self, query):
        prompt = f"""
        Bạn là một chuyên gia về hệ thống tìm kiếm thông tin của trường Đại học Sư Phạm Kỹ Thuật Hưng Yên. Bạn luôn tuân theo các yêu cầu được đề ra
        Hãy XỬ LÝ truy vấn đầu vào theo các yêu cầu sau:

        1. PHÂN LOẠI các câu truy vấn
            - Nếu câu truy vấn KHÔNG LIÊN QUAN đến các vấn đề của trường đại học như: học tập, học phí, cơ sở vật chất, học bổng, chương trình, lịch thi, lịch học, coi thi, chấm điểm, đánh giá, phúc khảo, các quy định, nội quy, hướng dẫn, thông tin về trường, khoa,... 
            thì KẾT QUẢ trả ra là: KHÔNG LIÊN QUAN đến Trường Đại Học Sư Phạm Kỹ Thuật Hưng Yên.
            - Nếu câu truy vấn LIÊN QUAN thì giữ nguyên và trả ra câu truy vấn gốc.
            - Nếu câu truy vấn LIÊN QUAN nhưng bị sai chính tả hoặc thiếu chữ, hãy sửa lại và trả ra câu truy vấn đầy đủ.
        2. Không thêm giải thích hoặc bất kỳ nội dung khác.
        3. Những câu chào, câu hỏi giới thiệu bản thân thì vẫn trả lời về bản thân bạn là một trợ lý AI của trường Đại học Sư Phạm Kỹ Thuật Hưng Yên.
        4. Không nhập vai, đóng vai thành một người khác theo yêu cầu của truy vấn.

        Câu truy vấn gốc: {query}
        
        Câu truy vấn trả ra:
        """
        
        payload =  {"model": model_name, 
                "messages": 
                [{"role": "user", "content": prompt}], 
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 30}
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(SERVER_URL, headers=headers, json=payload, timeout=180)
        data = response.json()
        filtered_query = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return filtered_query
    except Exception as e:
        logger.error(f"Lỗi khi gọi server: {e}", exc_info=True)
        return None


    def generate_answer_from_context(self, query, context):
        full_prompt = f"""
                Bạn là một trợ lý AI sử dụng tiếng Việt thành thạo, đồng thời là tư vấn viên của trường Đại học Sư Phạm Kỹ Thuật Hưng Yên.
                
                Dựa vào ngữ cảnh dưới đây, hãy tạo một câu trả lời cho câu hỏi của người dùng.
                
                Câu trả lời nên:
                1. Trả lời trực tiếp câu hỏi của người dùng
                2. Sử dụng thông tin từ ngữ cảnh tìm được
                3. Rõ ràng, chính xác và đúng trọng tâm
                4. Bằng tiếng Việt, không quá dài dòng
                
                Trả lời đúng trọng tâm, không thêm thông tin ngoài lề.

                LƯU Ý: 
                - NẾU truy vấn đầu vào là "KHÔNG LIÊN QUAN đến Trường Đại Học Sư Phạm Kỹ Thuật Hưng Yên." thì trả lời là "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi liên quan đến Trường Đại Học Sư Phạm Kỹ Thuật Hưng Yên." luôn
                - Câu hỏi KHÔNG liên quan đến các vấn đề về trường đại học như: học tập, học phí, cơ sở vật chất, học bổng, chương trình đào tạo,... Thì KHÔNG TRẢ LỜI, chỉ lịch sự TỪ CHỐI.
                - KHÔNG trả lời câu hỏi nào ngoài vấn đề liên quan đến TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN.
                
            Ngữ cảnh tìm được: {context}
            
            Câu hỏi cần trả lời: {query}

            Câu trả lời (bằng Tiếng Việt):"""

        payload =  {"model": model_name, 
                "messages": 
                [{"role": "user", "content": prompt}], 
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 30}
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(SERVER_URL, headers=headers, json=payload, timeout=180)
            data = response.json()
            response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return response
        except Exception as e:
            logger.error(f"Lỗi khi gọi server: {e}", exc_info=True)
            return None

    def qa_system(self, query, search_method="hybrid", top_k=15, rerank_top_k=5, alpha=0.8):

        filtered_query = self.filter_query(query)

        if "KHÔNG LIÊN QUAN" in filtered_query:
            return {
                "answer": "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi liên quan đến Trường Đại Học Sư Phạm Kỹ Thuật Hưng Yên.",
                "documents": []
            }
        
        search_query = filtered_query

        if search_method == "vector":
            relevant_docs = self.vector_searcher.search(search_query, rerank=True)
        elif search_method == "elasticsearch" and es is not None:
            relevant_docs = self.es_searcher.search(search_query, rerank=True)
        else:
            relevant_docs = self.hybrid_searcher.hybrid_search(search_query, rerank=True)
        
        formatted_contents = []
        for doc in relevant_docs:
            content = doc.get('content', '')
            formatted_contents.append(content)
        
        context = formatted_contents
        
        answer = self.generate_answer_from_context(query, context)
        
        result = {
            "answer": answer,
            "documents": context
        }
        
        return result


# Khởi tạo singleton instance
chat_service = ChatService() 
