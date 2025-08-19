"""Qiita Semantic Search Tool for AI-PoC-Agents-v2."""

import json
import time
import re
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class QiitaSemanticSearchTool:
    """QiitaのAPIを活用したセマンティック検索ツール"""
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize Qiita search tool.
        
        Args:
            access_token: Optional Qiita API access token for higher rate limits
        """
        self.access_token = access_token
        self.base_url = "https://qiita.com/api/v2"
        self.session = requests.Session()
        
        # Set up headers
        if self.access_token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}"
            })
        
        # Initialize embedding model for semantic search
        try:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
            self.semantic_search_enabled = True
            logger.info("Semantic search model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load semantic search model: {e}")
            self.semantic_search_enabled = False
    
    def search_relevant_articles(self, 
                               project_theme: str, 
                               technical_keywords: List[str] = None,
                               max_articles: int = 30) -> List[Dict[str, Any]]:
        """
        プロジェクトテーマに関連するQiita記事を検索
        
        Args:
            project_theme: プロジェクトのテーマ
            technical_keywords: 技術キーワードのリスト
            max_articles: 取得する最大記事数
            
        Returns:
            関連記事のリスト
        """
        logger.info(f"Searching Qiita articles for theme: {project_theme}")
        
        technical_keywords = technical_keywords or []
        
        try:
            # 1. キーワードベース検索
            keyword_results = self._keyword_search(project_theme, technical_keywords)
            logger.info(f"Found {len(keyword_results)} articles from keyword search")
            
            # 2. セマンティック検索による品質向上
            if self.semantic_search_enabled and keyword_results:
                semantic_results = self._semantic_search(project_theme, keyword_results)
                logger.info("Applied semantic ranking to articles")
            else:
                semantic_results = keyword_results
            
            # 3. 記事品質フィルタリング
            filtered_articles = self._filter_high_quality_articles(semantic_results)
            logger.info(f"Filtered to {len(filtered_articles)} high-quality articles")
            
            return filtered_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Error searching Qiita articles: {e}")
            return []
    
    def _keyword_search(self, theme: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Qiita API v2を使用したキーワード検索（Pythonコード重視）"""
        all_articles = []
        
        # Pythonコード実装に特化したクエリ
        python_focused_queries = [
            f"{theme} Python 実装",
            f"{theme} Python コード",
            f"{theme} Python サンプル",
            f"{theme} Python 開発",
            f"Python {theme}",
            theme  # 基本テーマも含める
        ]
        
        # 技術キーワードとPythonの組み合わせ
        for keyword in keywords[:3]:
            python_focused_queries.extend([
                f"{theme} Python {keyword}",
                f"Python {keyword} {theme}",
                f"{theme} {keyword} 実装"
            ])
        
        # Pythonライブラリ・フレームワーク特化クエリ
        python_libraries = ['numpy', 'pandas', 'opencv', 'tensorflow', 'pytorch', 'scikit-learn', 'flask', 'django', 'streamlit']
        for lib in python_libraries[:5]:
            python_focused_queries.append(f"{theme} {lib}")
        
        for query in python_focused_queries:
            try:
                response = self.session.get(
                    f"{self.base_url}/items",
                    params={
                        "query": query,
                        "page": 1,
                        "per_page": 20
                    }
                )
                
                if response.status_code == 200:
                    articles = response.json()
                    all_articles.extend(articles)
                    logger.debug(f"Query '{query}' returned {len(articles)} articles")
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded, waiting...")
                    time.sleep(5.0)
                else:
                    logger.warning(f"API request failed with status {response.status_code}")
                    
                # レート制限対応
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in keyword search for '{query}': {e}")
                continue
        
        # 重複除去
        unique_articles = {}
        for article in all_articles:
            unique_articles[article['id']] = article
        
        return list(unique_articles.values())
    
    def _semantic_search(self, theme: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """セマンティック類似度による記事ランキング"""
        if not self.semantic_search_enabled or not articles:
            return articles
        
        try:
            # プロジェクトテーマのベクトル化
            theme_embedding = self.embedding_model.encode([theme])[0]
            
            # 記事コンテンツのベクトル化
            article_texts = []
            for article in articles:
                # タイトル + タグ + 本文の一部を結合
                content = f"{article['title']} "
                content += " ".join([tag['name'] for tag in article.get('tags', [])])
                
                # 本文は最初の1000文字のみ使用（処理効率化）
                body = article.get('body', '')
                if len(body) > 1000:
                    body = body[:1000]
                content += f" {body}"
                
                article_texts.append(content)
            
            if not article_texts:
                return articles
                
            article_embeddings = self.embedding_model.encode(article_texts)
            
            # コサイン類似度計算
            similarities = cosine_similarity([theme_embedding], article_embeddings)[0]
            
            # 類似度順にソート
            scored_articles = []
            for i, article in enumerate(articles):
                article_copy = article.copy()
                article_copy['semantic_similarity'] = float(similarities[i])
                scored_articles.append(article_copy)
            
            return sorted(scored_articles, key=lambda x: x.get('semantic_similarity', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return articles
    
    def _filter_high_quality_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """高品質記事のフィルタリング（Pythonコード重視）"""
        filtered = []
        
        for article in articles:
            quality_score = 0
            body = article.get('body', '')
            
            # Pythonコード含有率を重視
            python_code_indicators = [
                'python', 'import ', 'def ', 'class ', 'if __name__',
                'numpy', 'pandas', 'matplotlib', 'opencv', 'tensorflow', 'pytorch'
            ]
            python_code_score = sum(1 for indicator in python_code_indicators 
                                  if indicator.lower() in body.lower())
            if python_code_score >= 5:
                quality_score += 5  # Pythonコード豊富
            elif python_code_score >= 3:
                quality_score += 3
            elif python_code_score >= 1:
                quality_score += 1
            
            # Pythonコードブロック数を重視
            python_code_blocks = len(re.findall(r'```python|```py', body, re.IGNORECASE))
            if python_code_blocks >= 3:
                quality_score += 4
            elif python_code_blocks >= 1:
                quality_score += 2
            
            # 一般的なコードブロック
            code_blocks = len(re.findall(r'```', body)) // 2  # ペアで計算
            if code_blocks >= 3:
                quality_score += 2
            elif code_blocks >= 1:
                quality_score += 1
            
            # いいね数（LGTM数）
            likes_count = article.get('likes_count', 0)
            if likes_count >= 10:
                quality_score += 3
            elif likes_count >= 5:
                quality_score += 2
            elif likes_count >= 1:
                quality_score += 1
            
            # ストック数
            stocks_count = article.get('stocks_count', 0)
            if stocks_count >= 10:
                quality_score += 3
            elif stocks_count >= 3:
                quality_score += 2
            elif stocks_count >= 1:
                quality_score += 1
                
            # 記事の長さ（品質の指標として）
            body_length = len(body)
            if body_length >= 2000:
                quality_score += 2
            elif body_length >= 1000:
                quality_score += 1
            
            # コメント数
            comments_count = article.get('comments_count', 0)
            if comments_count >= 5:
                quality_score += 2
            elif comments_count >= 1:
                quality_score += 1
                
            # セマンティック類似度（利用可能な場合）
            semantic_similarity = article.get('semantic_similarity', 0)
            if semantic_similarity >= 0.5:
                quality_score += 3
            elif semantic_similarity >= 0.3:
                quality_score += 2
            elif semantic_similarity >= 0.1:
                quality_score += 1
            
            # Pythonタグが含まれているかチェック
            tags = [tag['name'].lower() for tag in article.get('tags', [])]
            if 'python' in tags:
                quality_score += 3
            
            # 機械学習・AI関連タグチェック
            ml_tags = ['machine-learning', 'ai', 'deep-learning', 'opencv', 'tensorflow', 'pytorch']
            ml_tag_count = sum(1 for tag in tags if any(ml_tag in tag for ml_tag in ml_tags))
            if ml_tag_count >= 2:
                quality_score += 2
            elif ml_tag_count >= 1:
                quality_score += 1
            
            # 品質閾値を満たす記事のみ（Pythonコード重視のため閾値を調整）
            if quality_score >= 5:  # Pythonコード重視で閾値上げ
                article['quality_score'] = quality_score
                article['python_code_score'] = python_code_score
                article['python_code_blocks'] = python_code_blocks
                filtered.append(article)
        
        # 品質スコア順にソート
        return sorted(filtered, key=lambda x: x.get('quality_score', 0), reverse=True)
    
    def extract_implementation_insights(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """記事から実装のインサイトを抽出（Pythonコード特化）"""
        insights = {
            'common_technologies': defaultdict(int),
            'python_libraries': defaultdict(int),
            'implementation_patterns': [],
            'code_examples': [],
            'python_code_examples': [],
            'article_summaries': []
        }
        
        for article in articles[:20]:  # 上位20記事を分析（Pythonコード重視のため増やす）
            try:
                content = article.get('body', '')
                
                # Python特化技術スタックの抽出
                python_tech_patterns = [
                    r'Python', r'numpy', r'pandas', r'matplotlib', r'seaborn', r'plotly',
                    r'opencv', r'cv2', r'PIL', r'Pillow', r'scikit-learn', r'sklearn',
                    r'tensorflow', r'keras', r'pytorch', r'torch', r'transformers',
                    r'flask', r'django', r'fastapi', r'streamlit', r'gradio',
                    r'jupyter', r'google colab', r'anaconda', r'pip', r'conda'
                ]
                
                for pattern in python_tech_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        clean_pattern = re.sub(r'\\\.', '.', pattern)
                        insights['python_libraries'][clean_pattern] += 1
                
                # 一般技術スタック
                general_tech_patterns = [
                    r'Docker', r'AWS', r'GCP', r'Azure', r'PostgreSQL', r'MySQL', 
                    r'Redis', r'MongoDB', r'JavaScript', r'React', r'Vue\.js'
                ]
                
                for pattern in general_tech_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        clean_pattern = re.sub(r'\\\.', '.', pattern)
                        insights['common_technologies'][clean_pattern] += 1
                
                # Pythonコード例の優先抽出
                python_code_blocks = re.findall(r'```(?:python|py)\n(.*?)\n```', content, re.DOTALL | re.IGNORECASE)
                for code in python_code_blocks[:3]:  # Pythonコードは多めに抽出
                    if len(code.strip()) > 30 and ('import' in code or 'def' in code or 'class' in code):
                        insights['python_code_examples'].append({
                            'code': code.strip()[:500],  # Pythonコードは長めに保存
                            'article_title': article['title'],
                            'article_url': article['url'],
                            'likes': article.get('likes_count', 0),
                            'python_code_score': article.get('python_code_score', 0),
                            'type': 'python_specific'
                        })
                
                # 一般的なコード例も抽出（Pythonコード以外）
                general_code_blocks = re.findall(r'```(?!python|py)[\w]*\n(.*?)\n```', content, re.DOTALL | re.IGNORECASE)
                for code in general_code_blocks[:1]:  # 一般コードは少なめ
                    if len(code.strip()) > 50:
                        insights['code_examples'].append({
                            'code': code.strip()[:300],
                            'article_title': article['title'],
                            'article_url': article['url'],
                            'likes': article.get('likes_count', 0),
                            'type': 'general'
                        })
                
                # 記事要約
                insights['article_summaries'].append({
                    'title': article['title'],
                    'url': article['url'],
                    'tags': [tag['name'] for tag in article.get('tags', [])],
                    'likes': article.get('likes_count', 0),
                    'stocks': article.get('stocks_count', 0),
                    'quality_score': article.get('quality_score', 0),
                    'semantic_similarity': article.get('semantic_similarity', 0),
                    'summary': content[:200] + '...' if len(content) > 200 else content
                })
                
            except Exception as e:
                logger.error(f"Error processing article {article.get('title', 'Unknown')}: {e}")
                continue
        
        # 頻度順ソート（Python特化）
        insights['python_libraries'] = dict(
            sorted(insights['python_libraries'].items(), key=lambda x: x[1], reverse=True)
        )
        insights['common_technologies'] = dict(
            sorted(insights['common_technologies'].items(), key=lambda x: x[1], reverse=True)
        )
        
        # PythonコードスコアでPythonコード例をソート
        insights['python_code_examples'] = sorted(
            insights['python_code_examples'], 
            key=lambda x: (x.get('python_code_score', 0), x.get('likes', 0)), 
            reverse=True
        )
        
        return insights
    
    def generate_poc_ideas_from_articles(self, 
                                       project_theme: str,
                                       articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Qiita記事からPoCアイデアを生成
        
        Args:
            project_theme: プロジェクトテーマ
            articles: Qiita記事のリスト
            
        Returns:
            生成されたPoCアイデアのリスト
        """
        if not articles:
            logger.warning("No articles provided for idea generation")
            return []
        
        # 実装インサイト抽出
        insights = self.extract_implementation_insights(articles)
        
        # 上位記事から実装パターンを抽出
        implementation_patterns = []
        for article in articles[:5]:
            tags = [tag['name'] for tag in article.get('tags', [])]
            pattern = {
                'title': article['title'],
                'technologies': [tag for tag in tags if any(tech.lower() in tag.lower() 
                                                          for tech in ['python', 'js', 'react', 'django', 'flask'])],
                'approach': article.get('body', '')[:500],
                'quality_metrics': {
                    'likes': article.get('likes_count', 0),
                    'stocks': article.get('stocks_count', 0),
                    'quality_score': article.get('quality_score', 0)
                }
            }
            implementation_patterns.append(pattern)
        
        # PoCアイデア生成（Python特化）
        poc_ideas = []
        
        # Pythonライブラリを使用したアイデア生成
        top_python_libraries = list(insights['python_libraries'].keys())[:5]
        
        for i, python_lib in enumerate(top_python_libraries[:3]):
            idea = {
                'id': f'qiita_python_idea_{i+1}',
                'title': f'{project_theme} using {python_lib}',
                'description': f'{project_theme}を{python_lib}で実装するPython PoC',
                'technical_approach': f'Python + {python_lib}',
                'inspiration_source': 'qiita_python_articles',
                'reference_articles': [
                    {
                        'title': article['title'],
                        'url': article['url'],
                        'relevance_score': article.get('semantic_similarity', 0),
                        'python_code_score': article.get('python_code_score', 0),
                        'python_code_blocks': article.get('python_code_blocks', 0)
                    }
                    for article in articles[:5] 
                    if python_lib.lower() in article.get('body', '').lower()
                ],
                'implementation_complexity': min(2 + i, 5),
                'expected_impact': max(5 - i, 2),
                'feasibility_score': 0.85 - (i * 0.05),  # Python実装は実現性高め
                'estimated_effort_hours': 12 + (i * 6),  # Python実装は効率的
                'python_code_examples': [
                    example for example in insights['python_code_examples']
                    if python_lib.lower() in example.get('code', '').lower()
                ][:3],
                'code_examples': [
                    example for example in insights['code_examples']
                    if python_lib.lower() in example.get('code', '').lower()
                ][:1]
            }
            poc_ideas.append(idea)
        
        # 一般的な技術スタックでの補完アイデア
        top_general_tech = list(insights['common_technologies'].keys())[:3]
        
        for i, tech_stack in enumerate(top_general_tech[:2]):
            if tech_stack not in top_python_libraries:  # 重複避け
                idea = {
                    'id': f'qiita_general_idea_{i+1}',
                    'title': f'{project_theme} with Python + {tech_stack}',
                    'description': f'{project_theme}をPythonと{tech_stack}で実装するPoC',
                    'technical_approach': f'Python + {tech_stack}',
                    'inspiration_source': 'qiita_mixed_articles',
                    'reference_articles': [
                        {
                            'title': article['title'],
                            'url': article['url'],
                            'relevance_score': article.get('semantic_similarity', 0)
                        }
                        for article in articles[:3] 
                        if tech_stack.lower() in article.get('body', '').lower()
                    ],
                    'implementation_complexity': min(3 + i, 5),
                    'expected_impact': max(4 - i, 2),
                    'feasibility_score': 0.75 - (i * 0.05),
                    'estimated_effort_hours': 20 + (i * 8),
                    'code_examples': [
                        example for example in insights['code_examples']
                        if tech_stack.lower() in example.get('code', '').lower()
                    ][:2]
                }
                poc_ideas.append(idea)
        
        # 実装パターンベースのアイデア
        for i, pattern in enumerate(implementation_patterns[:2]):
            if pattern['technologies']:
                idea = {
                    'id': f'qiita_pattern_{i+1}',
                    'title': f'{project_theme} inspired by "{pattern["title"][:50]}..."',
                    'description': f'{pattern["title"]}のアプローチを参考にした{project_theme}の実装',
                    'technical_approach': ', '.join(pattern['technologies'][:3]),
                    'inspiration_source': 'qiita_implementation_pattern',
                    'reference_articles': [{
                        'title': pattern['title'],
                        'technologies': pattern['technologies']
                    }],
                    'implementation_complexity': 3,
                    'expected_impact': 4,
                    'feasibility_score': 0.75,
                    'estimated_effort_hours': 20,
                    'inspiration_notes': pattern['approach'][:200]
                }
                poc_ideas.append(idea)
        
        logger.info(f"Generated {len(poc_ideas)} PoC ideas from Qiita articles")
        return poc_ideas
    
    def enhance_existing_ideas(self, 
                             existing_ideas: List[Dict[str, Any]], 
                             project_theme: str) -> List[Dict[str, Any]]:
        """
        既存のアイデアをQiita情報で強化
        
        Args:
            existing_ideas: 既存のPoCアイデア
            project_theme: プロジェクトテーマ
            
        Returns:
            強化されたアイデアのリスト
        """
        if not existing_ideas:
            return existing_ideas
        
        # 関連記事検索
        keywords = []
        for idea in existing_ideas:
            if isinstance(idea, dict):
                keywords.extend([
                    idea.get('title', ''),
                    idea.get('technical_approach', ''),
                    idea.get('name', '')
                ])
            elif hasattr(idea, 'title'):
                keywords.extend([idea.title, getattr(idea, 'technical_approach', '')])
        
        # 重複除去とフィルタリング
        keywords = list(set([k for k in keywords if k and len(k) > 2]))
        
        articles = self.search_relevant_articles(project_theme, keywords[:5])
        
        if not articles:
            logger.warning("No relevant articles found for idea enhancement")
            return existing_ideas
        
        insights = self.extract_implementation_insights(articles)
        
        # 既存アイデアを強化
        enhanced_ideas = []
        for idea in existing_ideas:
            if isinstance(idea, dict):
                enhanced_idea = idea.copy()
            else:
                enhanced_idea = idea.__dict__.copy() if hasattr(idea, '__dict__') else {}
            
            # 技術スタック推奨
            if insights['common_technologies']:
                top_tech = list(insights['common_technologies'].keys())[:3]
                enhanced_idea['qiita_recommended_technologies'] = top_tech
                enhanced_idea['technology_trends'] = dict(list(insights['common_technologies'].items())[:5])
            
            # 実装参考記事
            idea_keywords = [
                enhanced_idea.get('title', ''),
                enhanced_idea.get('name', ''),
                enhanced_idea.get('technical_approach', '')
            ]
            
            relevant_articles = []
            for article in articles[:5]:
                relevance_score = 0
                article_content = f"{article['title']} {article.get('body', '')[:500]}".lower()
                
                for keyword in idea_keywords:
                    if keyword and keyword.lower() in article_content:
                        relevance_score += 1
                
                if relevance_score > 0:
                    relevant_articles.append({
                        'title': article['title'],
                        'url': article['url'],
                        'relevance_score': relevance_score,
                        'quality_metrics': {
                            'likes': article.get('likes_count', 0),
                            'stocks': article.get('stocks_count', 0)
                        }
                    })
            
            enhanced_idea['qiita_reference_articles'] = sorted(
                relevant_articles, key=lambda x: x['relevance_score'], reverse=True
            )[:3]
            
            # 実装例コード
            relevant_codes = []
            for example in insights['code_examples'][:3]:
                code_relevance = sum(1 for keyword in idea_keywords 
                                   if keyword and keyword.lower() in example.get('code', '').lower())
                if code_relevance > 0:
                    relevant_codes.append(example)
            
            enhanced_idea['qiita_code_examples'] = relevant_codes
            
            # メタデータ追加
            enhanced_idea['qiita_enhancement_applied'] = True
            enhanced_idea['qiita_articles_analyzed'] = len(articles)
            
            enhanced_ideas.append(enhanced_idea)
        
        logger.info(f"Enhanced {len(enhanced_ideas)} ideas with Qiita insights")
        return enhanced_ideas


if __name__ == "__main__":
    """動作確認用のメイン関数"""
    import argparse
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Test Qiita Semantic Search Tool")
    parser.add_argument("--theme", type=str, default="OCR文字認識", help="Project theme to search for")
    parser.add_argument("--keywords", type=str, nargs="*", default=["Python", "OpenCV", "機械学習"], help="Technical keywords")
    parser.add_argument("--max-articles", type=int, default=10, help="Maximum articles to retrieve")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic search")
    
    args = parser.parse_args()
    
    print("=== Qiita Semantic Search Tool Test ===")
    print(f"Theme: {args.theme}")
    print(f"Keywords: {args.keywords}")
    print(f"Max articles: {args.max_articles}")
    print(f"Semantic search: {'Disabled' if args.no_semantic else 'Enabled'}")
    print()
    
    # Initialize tool
    qiita_access_token = os.getenv("QIITA_ACCESS_TOKEN")
    if qiita_access_token:
        print(f"✓ Using Qiita access token (length: {len(qiita_access_token)})")
    else:
        print("⚠ No Qiita access token - using anonymous access (limited rate limit)")
    
    tool = QiitaSemanticSearchTool(access_token=qiita_access_token)
    
    if args.no_semantic:
        tool.semantic_search_enabled = False
    
    try:
        # Step 1: Search for relevant articles
        print("\n1. Searching for relevant articles...")
        articles = tool.search_relevant_articles(
            project_theme=args.theme,
            technical_keywords=args.keywords,
            max_articles=args.max_articles
        )
        
        print(f"Found {len(articles)} articles")
        
        if not articles:
            print("No articles found. Exiting.")
            exit(1)
        
        # Step 2: Display top articles
        print("\n2. Top articles found:")
        for i, article in enumerate(articles[:5], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   URL: {article['url']}")
            print(f"   Likes: {article.get('likes_count', 0)}")
            print(f"   Stocks: {article.get('stocks_count', 0)}")
            print(f"   Comments: {article.get('comments_count', 0)}")
            print(f"   Quality Score: {article.get('quality_score', 'N/A')}")
            if 'semantic_similarity' in article:
                print(f"   Semantic Similarity: {article['semantic_similarity']:.3f}")
            
            tags = [tag['name'] for tag in article.get('tags', [])]
            if tags:
                print(f"   Tags: {', '.join(tags[:5])}")
            
            # Show first 200 characters of body
            body = article.get('body', '')
            if body:
                preview = body.replace('\n', ' ')[:200]
                print(f"   Preview: {preview}{'...' if len(body) > 200 else ''}")
        
        # Step 3: Extract implementation insights
        print("\n3. Extracting implementation insights...")
        insights = tool.extract_implementation_insights(articles)
        
        print(f"\nTop Python libraries found:")
        for lib, count in list(insights['python_libraries'].items())[:8]:
            print(f"   {lib}: {count} articles")
        
        print(f"\nTop general technologies found:")
        for tech, count in list(insights['common_technologies'].items())[:5]:
            print(f"   {tech}: {count} articles")
        
        print(f"\nPython code examples found: {len(insights['python_code_examples'])}")
        for i, example in enumerate(insights['python_code_examples'][:3], 1):
            print(f"\n   Python Example {i} from '{example['article_title']}':")
            print(f"   Python Code Score: {example.get('python_code_score', 0)}")
            print(f"   Code: {example['code'][:150]}{'...' if len(example['code']) > 150 else ''}")
        
        print(f"\nGeneral code examples found: {len(insights['code_examples'])}")
        for i, example in enumerate(insights['code_examples'][:2], 1):
            print(f"\n   General Example {i} from '{example['article_title']}':")
            print(f"   Code: {example['code'][:100]}{'...' if len(example['code']) > 100 else ''}")
        
        print(f"\nArticle summaries: {len(insights['article_summaries'])}")
        
        # Step 4: Generate PoC ideas
        print("\n4. Generating PoC ideas from articles...")
        poc_ideas = tool.generate_poc_ideas_from_articles(args.theme, articles)
        
        print(f"\nGenerated {len(poc_ideas)} PoC ideas:")
        for i, idea in enumerate(poc_ideas, 1):
            print(f"\n{i}. {idea['title']}")
            print(f"   Description: {idea['description']}")
            print(f"   Technical Approach: {idea['technical_approach']}")
            print(f"   Feasibility Score: {idea['feasibility_score']:.2f}")
            print(f"   Expected Impact: {idea['expected_impact']}")
            print(f"   Estimated Effort: {idea['estimated_effort_hours']} hours")
            print(f"   Inspiration Source: {idea['inspiration_source']}")
            
            if idea['reference_articles']:
                print(f"   Reference Articles: {len(idea['reference_articles'])}")
                for ref in idea['reference_articles'][:2]:
                    print(f"     - {ref.get('title', 'Unknown title')}")
            
            if idea.get('code_examples'):
                print(f"   Code Examples: {len(idea['code_examples'])}")
        
        # Step 5: Test idea enhancement
        print("\n5. Testing idea enhancement...")
        sample_ideas = [
            {
                'title': f'{args.theme}の基本実装',
                'description': f'{args.theme}をPythonで実装する基本的なアプローチ',
                'technical_approach': 'Python'
            },
            {
                'title': f'{args.theme}のWebアプリ版',
                'description': f'Web ブラウザで使える{args.theme}システム',
                'technical_approach': 'Flask, JavaScript'
            }
        ]
        
        enhanced_ideas = tool.enhance_existing_ideas(sample_ideas, args.theme)
        
        print(f"\nEnhanced {len(enhanced_ideas)} existing ideas:")
        for i, idea in enumerate(enhanced_ideas, 1):
            print(f"\n{i}. {idea['title']}")
            
            if idea.get('qiita_recommended_technologies'):
                print(f"   Recommended Technologies: {', '.join(idea['qiita_recommended_technologies'][:3])}")
            
            if idea.get('qiita_reference_articles'):
                print(f"   Reference Articles: {len(idea['qiita_reference_articles'])}")
                for ref in idea['qiita_reference_articles'][:2]:
                    print(f"     - {ref['title']} (Score: {ref['relevance_score']})")
            
            if idea.get('qiita_code_examples'):
                print(f"   Code Examples: {len(idea['qiita_code_examples'])}")
        
        print("\n✅ Qiita Semantic Search Tool test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)