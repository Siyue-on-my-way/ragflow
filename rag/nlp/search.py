# -*- coding: utf-8 -*-
import json
import re
from copy import deepcopy

from pathlib import Path
from openai import OpenAI
import xml.etree.ElementTree as ET
import logging
from elasticsearch_dsl import Q, Search
from typing import List, Optional, Dict, Union
from dataclasses import dataclass

from rag.settings import es_logger
from rag.utils import rmSpace
from rag.nlp import rag_tokenizer, query
import numpy as np

def index_name(uid): return f"ragflow_{uid}"

class Dealer:
    def __init__(self, es):
        self.qryr = query.EsQueryer(es)
        self.qryr.flds = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "content_ltks^2",
            "content_sm_ltks"]
        self.es = es

    @dataclass
    class SearchResult:
        total: int
        ids: List[str]
        query_vector: List[float] = None
        field: Optional[Dict] = None
        highlight: Optional[Dict] = None
        aggregation: Union[List, Dict, None] = None
        keywords: Optional[List[str]] = None
        group_docs: List[List] = None

    def _vector(self, txt, emb_mdl, sim=0.8, topk=10):
        qv, c = emb_mdl.encode_queries(txt)
        return {
            "field": "q_%d_vec" % len(qv),
            "k": topk,
            "similarity": sim,
            "num_candidates": topk * 2,
            "query_vector": [float(v) for v in qv]
        }

    def search(self, req, idxnm, emb_mdl=None):
        qst = req.get("question", "")
        bqry, keywords = self.qryr.question(qst)
        def add_filters(bqry):
            nonlocal req
            if req.get("kb_ids"):
                bqry.filter.append(Q("terms", kb_id=req["kb_ids"]))
            if req.get("doc_ids"):
                bqry.filter.append(Q("terms", doc_id=req["doc_ids"]))
            if "available_int" in req:
                if req["available_int"] == 0:
                    bqry.filter.append(Q("range", available_int={"lt": 1}))
                else:
                    bqry.filter.append(
                        Q("bool", must_not=Q("range", available_int={"lt": 1})))
            return bqry

        bqry = add_filters(bqry)
        bqry.boost = 0.05

        s = Search()
        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        src = req.get("fields", ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd",
                                 "image_id", "doc_id", "q_512_vec", "q_768_vec", "position_int",
                                 "q_1024_vec", "q_1536_vec", "available_int", "content_with_weight"])

        s = s.query(bqry)[pg * ps:(pg + 1) * ps]
        s = s.highlight("content_ltks")
        s = s.highlight("title_ltks")
        if not qst:
            if not req.get("sort"):
                s = s.sort(
                    {"create_time": {"order": "desc", "unmapped_type": "date"}},
                    {"create_timestamp_flt": {
                        "order": "desc", "unmapped_type": "float"}}
                )
            else:
                s = s.sort(
                    {"page_num_int": {"order": "asc", "unmapped_type": "float",
                                      "mode": "avg", "numeric_type": "double"}},
                    {"top_int": {"order": "asc", "unmapped_type": "float",
                                 "mode": "avg", "numeric_type": "double"}},
                    {"create_time": {"order": "desc", "unmapped_type": "date"}},
                    {"create_timestamp_flt": {
                        "order": "desc", "unmapped_type": "float"}}
                )

        if qst:
            s = s.highlight_options(
                fragment_size=120,
                number_of_fragments=5,
                boundary_scanner_locale="zh-CN",
                boundary_scanner="SENTENCE",
                boundary_chars=",./;:\\!()，。？：！……（）——、"
            )
        s = s.to_dict()
        q_vec = []
        if req.get("vector"):
            assert emb_mdl, "No embedding model selected"
            s["knn"] = self._vector(
                qst, emb_mdl, req.get(
                    "similarity", 0.1), topk)
            s["knn"]["filter"] = bqry.to_dict()
            if "highlight" in s:
                del s["highlight"]
            q_vec = s["knn"]["query_vector"]
        es_logger.info("【Q】: {}".format(json.dumps(s)))
        res = self.es.search(deepcopy(s), idxnm=idxnm, timeout="600s", src=src)
        es_logger.info("TOTAL: {}".format(self.es.getTotal(res)))
        if self.es.getTotal(res) == 0 and "knn" in s:
            bqry, _ = self.qryr.question(qst, min_match="10%")
            bqry = add_filters(bqry)
            s["query"] = bqry.to_dict()
            s["knn"]["filter"] = bqry.to_dict()
            s["knn"]["similarity"] = 0.17
            res = self.es.search(s, idxnm=idxnm, timeout="600s", src=src)
            es_logger.info("【Q】: {}".format(json.dumps(s)))

        kwds = set([])
        for k in keywords:
            kwds.add(k)
            for kk in rag_tokenizer.fine_grained_tokenize(k).split(" "):
                if len(kk) < 2:
                    continue
                if kk in kwds:
                    continue
                kwds.add(kk)

        aggs = self.getAggregation(res, "docnm_kwd")

        return self.SearchResult(
            total=self.es.getTotal(res),
            ids=self.es.getDocIds(res),
            query_vector=q_vec,
            aggregation=aggs,
            highlight=self.getHighlight(res),
            field=self.getFields(res, src),
            keywords=list(kwds)
        )

    def text_to_sql(self, qst):
        client = OpenAI(
            api_key="sk-48cac4f6e9ad4dabbcc0688654e70820",  # 替换成真实DashScope的API_KEY
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
        )
        prompt = f"""
                你是一个es搜索的专家。你需要这对以下表的字段结构，根据用户的问题列表，写出最后一个问题的es的对应字段的过滤条件（如patent_id(专利号，如2017108907222，YYYY开通),pub_id(公开(公告)号： 如CN100577011C，CN开头),patent_type(专利类型：发明专利、实用新型、外观设计)，patent_status(案件状态：已下证、未下证)，trading_status(交易状态：待交易/已预定)），
                日期相关的（pub_date，issue_date，apply_date,price）字段是gt(大于)，lt(小于)；
                模糊匹配的关键词短句放在keyword里，并且去除调不相关的词语,如一些比较长的，与专利专业内容无关的用途、价值说明等 ；公司或者个人名称相关的放在applicant里。
                表结构如下：
                <表结构>TABLE patent_demo_longtut (
                    patent_id VARCHAR(20) NOT NULL COMMENT '专利号，如2017108907222，YYYY开通',
                    pub_id VARCHAR(20) NOT NULL COMMENT '公开(公告)号： 如CN100577011C，CN开头',
                    title VARCHAR(255) NOT NULL COMMENT '专利名称：如一种XX制备方法、加工方法、机器物品等',
                    summary TEXT COMMENT '摘要：针对专利内容的摘要',
                    patent_type VARCHAR(50) COMMENT '专利类型：发明专利、实用新型、外观设计',
                    patent_status VARCHAR(50) COMMENT '案件状态：已下证、未下证',
                    applicant VARCHAR(255) COMMENT '申请人：公司、学校名称或个人名字',
                    pub_date DATE COMMENT '发布日：如2021-01-01',
                    issue_date DATE COMMENT '发证日：如2022-01-01',
                    apply_date DATE COMMENT '申请日：如2021-01-01',
                    trading_status VARCHAR(50) COMMENT '交易状态：待交易/已预定',
                    price int COMMENT '价格：单位元',
                    manual_specification TEXT COMMENT '说明：如专利权维持，不提供发明人证件这样的说明',
                    PRIMARY KEY (patent_id) -- 假设专利号作为主键
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='专利信息表';
                </表结构>
                要求：1、确保生成的过滤条件是枚举值里的，不能随意添加。比如交易状态只有待交易和已预订两个类型,如果两个类型都有, 则不用添加这个过滤条件。
                2、如果输入有超过两组要求，则合并成一组或者挑选其中一组来形成输出。
                3、模糊匹配的关键词短句放在keyword里，并且去除调不相关的词语，如一些比较长的，与专利专业内容无关的用途、价值说明等
                4、价格相关，一般是超过1千的，如果没加单位，则默认是万为单位
                下面一些例子
                输入：
                <input>
                已下证 可以交易 2023年发布的 拐杖相关发明专利
                </input>    
                输出：
                <output> <patent_status>已下证</patent_status><trading_status>待交易</trading_status> <pub_date><gt>2023-01-01</gt><lt>2023-12-31</lt></pub_date><keyword> 拐杖</keyword><patent_type>发明专利</patent_type></output>

                输入：
                <input>
                教育笔记本（要有一定价值的，有制造需求，明年打算申报高企也想用）
                </input>
                <output><keyword>教育 笔记本</keyword></output>

                输入：
                <input>
                需要软件相关的专利 下证未下证，发明或实用，都行
                </input>  
                <output><keyword>软件</keyword></output>
                这里下证未下证都行就不加patent_status过滤，发明专利或者实用专利都行就不加patent_type过滤

                输入
                <input>
                求购电池热管理相关专利，发明专利或实用专利都行，不要未下证的，价格在2w以下均可接受 大模型结果
                </input>  
                <output><keyword>电池 热管理</keyword><patent_status>已下证</patent_status><price><lte>20000</lte></price></output>
                这里 发明专利或者实用专利都行就不加patent_type过滤
    
                输入：
                <input>
                    需要软件相关的专利 为下证
                </input>  
                <output><keyword>软件</keyword><patent_status>未下证</patent_status></output>
                这里未下证变成了错别字“为下证”，要纠正为 未下证 过滤

                输入：
                <input>
                请提供专利号是2018103108800的相关信息，包括是否已经下证、价格、专利类型
                </input>
                <output><patent_id>2018103108800</patent_id></output>
                这里是问某篇patent_id专利的相关信息，则这些询问的相关信息如patent_type、price和patent_status就不能加入到过滤条件里
            
                """
        completion = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {
                'role': 'system',
                'content': prompt
            },
            {
                'role': 'user',
                'content': f"""
                用户的输入
                <input>
                    {qst}
                </input>
                请直接给出输出
                """
            },
        ],
        temperature=0.09,
        stream=False)
        text = completion.choices[0].message.content
        es_logger.info(text)
        start_tag = '<output>'
        end_tag = '</output>'
        start_pos = text.find(start_tag)
        end_pos = text.find(end_tag) + len(end_tag)
        xml_data = text[start_pos:end_pos]
        try:
            root = ET.fromstring(xml_data)
        except Exception as e:
            es_logger.info(e)
            return None,text
        return root, text

  # 映射列名到处理函数或字段名
    def search_patent(self, req, idxnm, emb_mdl=None):
        qst = req.get("question", "")
        root, text = self.text_to_sql(qst)
        es_logger.info(f"{qst} 大模型结果 {text}")
        qst = ''
        if root:
            for column in ['keyword','title','summary']:
                for item in root.findall(column):
                    qst+= item.text + ' '
        bqry, keywords = self.qryr.question(qst)

        def add_filter_by_column(bqry, column, element, mapping):
          """根据列名和元素值添加过滤条件"""
          action = mapping.get(column)
          if action:
              if callable(action):
                  action(bqry, element)
              else:
                  # field_with_keyword = f"{action}.keyword" if isinstance(element, str) else action
                  bqry.filter.append(Q("term", **{action: element}))
                  
        def add_text_match_filter(bqry, column, element):
            es_logger.info(bqry)
            """根据列名和元素值添加模糊文本匹配过滤条件"""
            # 构造带有模糊匹配的Match查询
            text_match = Q("match", **{
                column: {
                    "query": element.text,
                    "fuzziness": "AUTO"
                }
            })
            
            # 检查bqry是否有should属性
            if hasattr(bqry, 'should'):
                # 如果有should属性，则直接添加should_clause
                bqry.should.append(text_match)
            else:
                # 如果没有should属性，创建一个新的should列表并添加should_clause
                bqry = Q("bool", must=bqry.must, should=[text_match])
            
            return bqry

        def handle_range_condition(bqry, column, element):
            """处理日期或价格的范围条件"""
            res = {}
            for child in element:
                if child.tag in ('lt', 'gt','lte','gte'):
                    res[child.tag] = child.text
            if res:
                bqry.filter.append(Q("range", **{column: res}))

        def add_text_to_sql(bqry, root):
            column_actions = {
              'patent_id': 'patent_id',
              'pub_id': 'pub_id',
              'applicant': add_text_match_filter,
              'manual_specification': 'manual_specification',
              'patent_type': 'patent_type',
              'patent_status': 'patent_status',
              'trading_status': 'trading_status',
              'pub_date': handle_range_condition,
              'issue_date': handle_range_condition,
              'apply_date': handle_range_condition,
              'price': handle_range_condition
            }
            for column in column_actions.keys():
              elements = root.findall(column)
              for element in elements:
                  action = column_actions[column]
                  if isinstance(action, str):  # 直接添加term过滤
                      add_filter_by_column(bqry, column, element.text, column_actions)
                  else:  # 调用处理函数
                      action(bqry, column,  element)
        if root:
            add_text_to_sql(bqry, root)

        def add_filters(bqry):
            nonlocal req
            if req.get("kb_ids"):
                bqry.filter.append(Q("terms", kb_id=req["kb_ids"]))
            if req.get("doc_ids"):
                bqry.filter.append(Q("terms", doc_id=req["doc_ids"]))
            if "available_int" in req:
                if req["available_int"] == 0:
                    bqry.filter.append(Q("range", available_int={"lt": 1}))
                else:
                    bqry.filter.append(
                        Q("bool", must_not=Q("range", available_int={"lt": 1})))
            return bqry

        bqry = add_filters(bqry)
        bqry.boost = 0.05

        s = Search()
        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        src = req.get("fields", ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd",
                                 "image_id", "doc_id", "q_512_vec", "q_768_vec", "position_int",
                                 "q_1024_vec", "q_1536_vec", "available_int", "content_with_weight"])

        s = s.query(bqry)[pg * ps:(pg + 1) * ps]
        s = s.highlight("content_ltks")
        s = s.highlight("title_ltks")
        if not qst:
            if not req.get("sort"):
                s = s.sort(
                    {"create_time": {"order": "desc", "unmapped_type": "date"}},
                    {"create_timestamp_flt": {
                        "order": "desc", "unmapped_type": "float"}}
                )
            else:
                s = s.sort(
                    {"page_num_int": {"order": "asc", "unmapped_type": "float",
                                      "mode": "avg", "numeric_type": "double"}},
                    {"top_int": {"order": "asc", "unmapped_type": "float",
                                 "mode": "avg", "numeric_type": "double"}},
                    {"create_time": {"order": "desc", "unmapped_type": "date"}},
                    {"create_timestamp_flt": {
                        "order": "desc", "unmapped_type": "float"}}
                )

        if qst:
            s = s.highlight_options(
                fragment_size=120,
                number_of_fragments=5,
                boundary_scanner_locale="zh-CN",
                boundary_scanner="SENTENCE",
                boundary_chars=",./;:\\!()，。？：！……（）——、"
            )
        s = s.to_dict()
        q_vec = []
        if req.get("vector"):
            assert emb_mdl, "No embedding model selected"
            s["knn"] = self._vector(
                qst, emb_mdl, req.get(
                    "similarity", 0.1), topk)
            s["knn"]["filter"] = bqry.to_dict()
            if "highlight" in s:
                del s["highlight"]
            q_vec = s["knn"]["query_vector"]
        es_logger.info("【Q】: {}".format(json.dumps(s)))
        res = self.es.search(deepcopy(s), idxnm=idxnm, timeout="600s", src=src)
        es_logger.info("TOTAL: {}".format(self.es.getTotal(res)))
        # if self.es.getTotal(res) == 0 and "knn" in s:
        #     bqry, _ = self.qryr.question(qst, min_match="10%")
        #     bqry = add_filters(bqry)
        #     s["query"] = bqry.to_dict()
        #     s["knn"]["filter"] = bqry.to_dict()
        #     s["knn"]["similarity"] = 0.17
        #     res = self.es.search(s, idxnm=idxnm, timeout="600s", src=src)
        #     es_logger.info("【Q】: {}".format(json.dumps(s)))

        kwds = set([])
        for k in keywords:
            kwds.add(k)
            for kk in rag_tokenizer.fine_grained_tokenize(k).split(" "):
                if len(kk) < 2:
                    continue
                if kk in kwds:
                    continue
                kwds.add(kk)

        aggs = self.getAggregation(res, "docnm_kwd")

        return self.SearchResult(
            total=self.es.getTotal(res),
            ids=self.es.getDocIds(res),
            query_vector=q_vec,
            aggregation=aggs,
            highlight=self.getHighlight(res),
            field=self.getFields(res, src),
            keywords=list(kwds)
        )

    def getAggregation(self, res, g):
        if not "aggregations" in res or "aggs_" + g not in res["aggregations"]:
            return
        bkts = res["aggregations"]["aggs_" + g]["buckets"]
        return [(b["key"], b["doc_count"]) for b in bkts]

    def getHighlight(self, res):
        def rmspace(line):
            eng = set(list("qwertyuioplkjhgfdsazxcvbnm"))
            r = []
            for t in line.split(" "):
                if not t:
                    continue
                if len(r) > 0 and len(
                        t) > 0 and r[-1][-1] in eng and t[0] in eng:
                    r.append(" ")
                r.append(t)
            r = "".join(r)
            return r

        ans = {}
        for d in res["hits"]["hits"]:
            hlts = d.get("highlight")
            if not hlts:
                continue
            ans[d["_id"]] = "".join([a for a in list(hlts.items())[0][1]])
        return ans

    def getFields(self, sres, flds):
        res = {}
        if not flds:
            return {}
        for d in self.es.getSource(sres):
            m = {n: d.get(n) for n in flds if d.get(n) is not None}
            for n, v in m.items():
                if isinstance(v, type([])):
                    m[n] = "\t".join([str(vv) if not isinstance(
                        vv, list) else "\t".join([str(vvv) for vvv in vv]) for vv in v])
                    continue
                if not isinstance(v, type("")):
                    m[n] = str(m[n])
                if n.find("tks") > 0:
                    m[n] = rmSpace(m[n])

            if m:
                res[d["id"]] = m
        return res

    @staticmethod
    def trans2floats(txt):
        return [float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        es_logger.info("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split(" ")
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr>0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(" "),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                es_logger.info("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" ##{c}$$"
                seted.add(c)

        return res, seted

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks"):
        _, keywords = self.qryr.question(query)
        ins_embd = [
            Dealer.trans2floats(
                sres.field[i].get("q_%d_vec" % len(sres.query_vector), "\t".join(["0"] * len(sres.query_vector)))) for i in sres.ids]
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split(" ")
            title_tks = [t for t in sres.field[i].get("title_tks", "").split(" ") if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)
        return sim, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks"):
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split(" ")
            title_tks = [t for t in sres.field[i].get("title_tks", "").split(" ") if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim,_ = rerank_mdl.similarity(" ".join(keywords), [rmSpace(" ".join(tks)) for tks in ins_tw])

        return tkweight*np.array(tksim) + vtweight*vtsim, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(" "),
                                           rag_tokenizer.tokenize(inst).split(" "))

    def retrieval(self, question, embd_mdl, tenant_id, kb_ids, page, page_size, similarity_threshold=0.2,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True, rerank_mdl=None):
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks
        req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "size": page_size,
               "question": question, "vector": True, "topk": top,
               "similarity": similarity_threshold,
               "available_int": 1}
        
        if tenant_id == 'longtut_test':
            sres = self.search_patent(req, tenant_id, embd_mdl)
        else:
            sres = self.search(req, index_name(tenant_id), embd_mdl)

        if rerank_mdl:
            sim, tsim, vsim = self.rerank_by_model(rerank_mdl,
                sres, question, 1 - vector_similarity_weight, vector_similarity_weight)
        else:
            sim, tsim, vsim = self.rerank(
                sres, question, 1 - vector_similarity_weight, vector_similarity_weight)
        idx = np.argsort(sim * -1)

        dim = len(sres.query_vector)
        start_idx = (page - 1) * page_size
        for i in idx:
            if sim[i] < similarity_threshold:
                break
            ranks["total"] += 1
            start_idx -= 1
            if start_idx >= 0:
                continue
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break
            id = sres.ids[i]
            dnm = sres.field[id]["docnm_kwd"]
            did = sres.field[id]["doc_id"]
            d = {
                "chunk_id": id,
                "content_ltks": sres.field[id]["content_ltks"],
                "content_with_weight": sres.field[id]["content_with_weight"],
                "doc_id": sres.field[id]["doc_id"],
                "docnm_kwd": dnm,
                "kb_id": sres.field[id]["kb_id"],
                "important_kwd": sres.field[id].get("important_kwd", []),
                "img_id": sres.field[id].get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": self.trans2floats(sres.field[id].get("q_%d_vec" % dim, "\t".join(["0"] * dim))),
                "positions": sres.field[id].get("position_int", "").split("\t")
            }
            if len(d["positions"]) % 5 == 0:
                poss = []
                for i in range(0, len(d["positions"]), 5):
                    poss.append([float(d["positions"][i]), float(d["positions"][i + 1]), float(d["positions"][i + 2]),
                                 float(d["positions"][i + 3]), float(d["positions"][i + 4])])
                d["positions"] = poss
            ranks["chunks"].append(d)
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
        ranks["doc_aggs"] = [{"doc_name": k,
                              "doc_id": v["doc_id"],
                              "count": v["count"]} for k,
                             v in sorted(ranks["doc_aggs"].items(),
                                         key=lambda x:x[1]["count"] * -1)]

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        from api.settings import chat_logger
        sql = re.sub(r"[ `]+", " ", sql)
        sql = sql.replace("%", "")
        es_logger.info(f"Get es sql: {sql}")
        replaces = []
        for r in re.finditer(r" ([a-z_]+_l?tks)( like | ?= ?)'([^']+)'", sql):
            fld, v = r.group(1), r.group(3)
            match = " MATCH({}, '{}', 'operator=OR;minimum_should_match=30%') ".format(
                fld, rag_tokenizer.fine_grained_tokenize(rag_tokenizer.tokenize(v)))
            replaces.append(
                ("{}{}'{}'".format(
                    r.group(1),
                    r.group(2),
                    r.group(3)),
                    match))

        for p, r in replaces:
            sql = sql.replace(p, r, 1)
        chat_logger.info(f"To es: {sql}")

        try:
            tbl = self.es.sql(sql, fetch_size, format)
            return tbl
        except Exception as e:
            chat_logger.error(f"SQL failure: {sql} =>" + str(e))
            return {"error": str(e)}

    def chunk_list(self, doc_id, tenant_id, max_count=1024, fields=["docnm_kwd", "content_with_weight", "img_id"]):
        s = Search()
        s = s.query(Q("match", doc_id=doc_id))[0:max_count]
        s = s.to_dict()
        es_res = self.es.search(s, idxnm=index_name(tenant_id), timeout="600s", src=fields)
        res = []
        for index, chunk in enumerate(es_res['hits']['hits']):
            res.append({fld: chunk['_source'].get(fld) for fld in fields})
        return res