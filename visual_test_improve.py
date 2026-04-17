"""
TransReID 篮球持球人 ReID 可视化验证
====================================
从 query 集随机抽取图片，在 gallery 中检索匹配，展示 Top-5 结果，
并标注匹配是否正确。运行 10 轮，每轮随机抽 1 张，结果保存为 HTML 报告。

用法：
  python visual_test.py --config_file configs/BallShow/vit_transreid_stride.yml \
      MODEL.DEVICE_ID "('0')" \
      TEST.WEIGHT 'logs/xxx/transformer_200.pth' \
      --num_rounds 10 --top_k 5
"""

import os
import sys
import time
import shutil
import argparse
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F

from config import cfg
from model import make_model
from datasets import make_dataloader
from utils.logger import setup_logger
from datasets.ballshow import BallShow
from datasets.bases import ImageDataset
import torchvision.transforms as T


def extract_all_features(model, device, dataset, cfg, num_query):
    """
    提取 query + gallery 所有图片的特征，返回结构化数据。
    dataset: BallShow 对象，含 .query 和 .gallery 列表
    """
    img_size = cfg.INPUT.SIZE_TEST
    val_transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    print(f"[特征提取] query: {len(dataset.query)} 张, gallery: {len(dataset.gallery)} 张")

    results = {
        'query': [],     # [(img_path, pid, camid, feat), ...]
        'gallery': [],   # [(img_path, pid, camid, feat), ...]
    }

    model.eval()
    with torch.no_grad():
        # 提取 query 特征
        for i, (img_path, pid, camid, trackid) in enumerate(dataset.query):
            img = val_transforms(ImageDataset._dummy_read(img_path)).unsqueeze(0).to(device)
            cam_label = torch.tensor([camid]).to(device)
            view_label = torch.tensor([trackid]).to(device)
            feat = model(img, cam_label=cam_label, view_label=view_label).cpu()
            results['query'].append((img_path, pid, camid, feat))

            if (i + 1) % 100 == 0:
                print(f"  query 进度: {i+1}/{len(dataset.query)}")

        # 提取 gallery 特征（分批处理加速）
        batch_size = 64
        gallery_paths, gallery_pids, gallery_camids, gallery_feats = [], [], [], []

        for i in range(0, len(dataset.gallery), batch_size):
            batch = dataset.gallery[i:i+batch_size]
            imgs = []
            cam_labels = []
            view_labels = []
            for img_path, pid, camid, trackid in batch:
                img = val_transforms(ImageDataset._dummy_read(img_path))
                imgs.append(img)
                cam_labels.append(camid)
                view_labels.append(trackid)

            imgs = torch.stack(imgs).to(device)
            cam_labels = torch.tensor(cam_labels).to(device)
            view_labels = torch.tensor(view_labels).to(device)
            feats = model(imgs, cam_label=cam_labels, view_label=view_labels).cpu()

            for j, (img_path, pid, camid, trackid) in enumerate(batch):
                gallery_paths.append(img_path)
                gallery_pids.append(pid)
                gallery_camids.append(camid)
                gallery_feats.append(feats[j])

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(dataset.gallery):
                print(f"  gallery 进度: {min(i+batch_size, len(dataset.gallery))}/{len(dataset.gallery)}")

        # 组装 gallery 结果
        for k in range(len(gallery_paths)):
            results['gallery'].append((gallery_paths[k], gallery_pids[k], gallery_camids[k], gallery_feats[k]))

    print(f"[特征提取] 完成！")
    return results


def _read_image_pil(img_path):
    """读取图片为 PIL Image"""
    from PIL import Image
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            pass
    return img


def search_top_k(query_feat, gallery_data, top_k=5):
    """
    在 gallery 中搜索与 query_feat 最相似的 top_k 个结果。
    返回: [(img_path, pid, camid, distance), ...] 按距离从小到大排序
    """
    # 构建 gallery 特征矩阵
    gf = torch.stack([item[3] for item in gallery_data], dim=0)  # [N, dim]
    gf_norm = F.normalize(gf, p=2, dim=1)
    # query_feat 可能是 1D [dim] 或 2D [1, dim]，统一处理
    if query_feat.dim() == 1:
        q_norm = F.normalize(query_feat, p=2, dim=0).unsqueeze(0)  # [1, dim]
    else:
        q_norm = F.normalize(query_feat, p=2, dim=1)  # [1, dim]

    # 余弦相似度 -> 距离
    similarity = torch.mm(q_norm, gf_norm.t()).squeeze(0)  # [N]
    distances = 1.0 - similarity

    # 排序取 top_k
    values, indices = torch.topk(distances, k=min(top_k, len(distances)), largest=False)

    results = []
    for v, idx in zip(values, indices):
        img_path, pid, camid, _ = gallery_data[idx]
        results.append((img_path, pid, camid, v.item()))
    return results


def generate_html_report(all_rounds, output_path, model_info, timestamp):
    """生成可视化 HTML 报告"""

    total = len(all_rounds)
    correct = sum(1 for r in all_rounds if r['top1_correct'])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>TransReID 可视化验证报告</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f5f5f5; color: #333; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
              color: white; padding: 30px 40px; text-align: center; }}
  .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
  .header .meta {{ font-size: 14px; opacity: 0.8; }}
  .stats {{ display: flex; justify-content: center; gap: 30px; padding: 20px;
            background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px auto;
            max-width: 600px; border-radius: 10px; }}
  .stat-item {{ text-align: center; }}
  .stat-item .num {{ font-size: 32px; font-weight: bold; }}
  .stat-item .num.green {{ color: #27ae60; }}
  .stat-item .num.red {{ color: #e74c3c; }}
  .stat-item .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
  .round {{ background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin: 20px auto; max-width: 1100px; padding: 24px; }}
  .round-title {{ font-size: 18px; font-weight: bold; margin-bottom: 16px;
                 padding-bottom: 10px; border-bottom: 2px solid #eee; display: flex;
                 align-items: center; gap: 10px; }}
  .round-title .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
                         font-size: 12px; font-weight: 600; }}
  .badge-correct {{ background: #d4edda; color: #155724; }}
  .badge-wrong {{ background: #f8d7da; color: #721c24; }}
  .query-section {{ margin-bottom: 16px; }}
  .query-label {{ font-size: 14px; font-weight: 600; color: #555; margin-bottom: 8px; }}
  .query-info {{ font-size: 13px; color: #777; margin-top: 4px; }}
  .img-row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }}
  .query-img-box {{ border: 3px solid #3498db; border-radius: 8px; overflow: hidden; flex-shrink: 0; }}
  .query-img-box img {{ display: block; height: 160px; width: auto; }}
  .results-grid {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .result-card {{ border: 2px solid #ddd; border-radius: 8px; overflow: hidden;
                  text-align: center; width: 140px; flex-shrink: 0; transition: transform 0.2s; }}
  .result-card:hover {{ transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
  .result-card.rank1-correct {{ border-color: #27ae60; }}
  .result-card.rank1-wrong {{ border-color: #e74c3c; }}
  .result-card img {{ display: block; height: 140px; width: auto; margin: 0 auto; }}
  .result-card .info {{ padding: 6px 4px; font-size: 11px; }}
  .result-card .rank {{ font-weight: bold; font-size: 13px; color: #333; }}
  .result-card .dist {{ color: #888; }}
  .result-card .match {{ font-size: 11px; margin-top: 2px; }}
  .match-yes {{ color: #27ae60; font-weight: 600; }}
  .match-no {{ color: #e74c3c; }}
  .footer {{ text-align: center; padding: 20px; color: #aaa; font-size: 12px; }}
</style>
</head>
<body>

<div class="header">
  <h1>TransReID 篮球持球人 ReID - 可视化验证报告</h1>
  <div class="meta">{model_info} | 生成时间: {timestamp} | 共 {total} 轮测试</div>
</div>

<div class="stats">
  <div class="stat-item">
    <div class="num">{total}</div>
    <div class="label">总测试轮数</div>
  </div>
  <div class="stat-item">
    <div class="num green">{correct}</div>
    <div class="label">Top-1 命中</div>
  </div>
  <div class="stat-item">
    <div class="num {'green' if correct/total >= 0.8 else 'red'}">{correct/total*100:.1f}%</div>
    <div class="label">Top-1 命中率</div>
  </div>
</div>

"""

    for i, rd in enumerate(all_rounds):
        status_class = "badge-correct" if rd['top1_correct'] else "badge-wrong"
        status_text = "Top-1 命中" if rd['top1_correct'] else "Top-1 未命中"

        # query 图片的相对路径（转成可显示的路径）
        q_path = rd['query_path']
        q_rel = _get_img_src(q_path)

        html += f"""
<div class="round">
  <div class="round-title">
    第 {i+1} 轮
    <span class="badge {status_class}">{status_text}</span>
  </div>
  <div class="query-section">
    <div class="query-label">Query 图片 (PID: {rd['query_pid']}, Camera: {rd['query_camid']})</div>
    <div class="img-row">
      <div class="query-img-box">
        <img src="{q_rel}" onerror="this.style.background='#eee';this.style.height='160px';this.style.width='100px';" />
      </div>
      <div style="padding-top:10px;">
        <div class="query-info">路径: {q_path}</div>
        <div class="query-info">真实 ID: {rd['query_pid']} | 摄像头: {rd['query_camid']}</div>
      </div>
    </div>
  </div>
  <div class="query-label" style="margin-top:12px;">Top-{rd['top_k']} 检索结果（按相似度从高到低）</div>
  <div class="results-grid">
"""
        for j, r in enumerate(rd['results']):
            r_path = r['path']
            r_rel = _get_img_src(r_path)
            is_correct = (r['pid'] == rd['query_pid'])
            is_junk = (r['pid'] == rd['query_pid'] and r['camid'] == rd['query_camid'])

            if j == 0:
                card_class = "rank1-correct" if is_correct else "rank1-wrong"
            else:
                card_class = ""
            match_class = "match-yes" if is_correct else "match-no"
            match_text = "匹配" if is_correct else "不匹配"
            if is_junk:
                match_text += " (同摄)"

            html += f"""
    <div class="result-card {card_class}">
      <img src="{r_rel}" onerror="this.style.background='#eee';this.style.height='140px';this.style.width='100px';" />
      <div class="info">
        <div class="rank">Rank-{j+1}</div>
        <div class="dist">距离: {r['distance']:.4f}</div>
        <div class="match {match_class}">{match_text}</div>
        <div style="color:#aaa;font-size:10px;">ID:{r['pid']} Cam:{r['camid']}</div>
      </div>
    </div>
"""

        html += """
  </div>
</div>
"""

    html += """
<div class="footer">
  TransReID Visual Verification Report | Auto-generated
</div>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[报告] 已保存至 {output_path}")


def _get_img_src(img_path):
    """
    将图片绝对路径转为 HTML 可用的路径。
    将图片复制到报告同级的 images/ 目录中，返回相对路径。
    """
    return img_path.replace('\\', '/')


def main():
    # 先从 sys.argv 中手动提取自定义参数，再交给 argparse 处理其余的
    import sys
    argv = list(sys.argv[1:])

    num_rounds = 10
    top_k = 5
    debug_limit = 0

    # 提取并移除自定义参数
    def extract_and_remove(args_list, flag):
        if flag in args_list:
            idx = args_list.index(flag)
            if idx + 1 < len(args_list):
                val = args_list[idx + 1]
                args_list.pop(idx)
                args_list.pop(idx)
                return val
        return None

    v = extract_and_remove(argv, '--num_rounds')
    if v is not None:
        num_rounds = int(v)
    v = extract_and_remove(argv, '--top_k')
    if v is not None:
        top_k = int(v)
    v = extract_and_remove(argv, '--debug_limit')
    if v is not None:
        debug_limit = int(v)

    # 用剩余 argv 构建 parser
    parser = argparse.ArgumentParser(description="TransReID Visual Test")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    print(f"[Config] num_rounds={num_rounds}, top_k={top_k}, debug_limit={debug_limit}")

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(f"Visual test config: {cfg}")

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cpu':
        print("Warning: Running on CPU, inference will be slow.")

    # 加载数据集（原始 BallShow 对象，用于获取 query/gallery 列表和路径）
    print("[Init] 加载数据集...")
    dataset = BallShow(root=cfg.DATASETS.ROOT_DIR)

    # 调试模式：截断 query/gallery 快速复现报错
    if debug_limit > 0:
        dataset.query = dataset.query[:debug_limit]
        dataset.gallery = dataset.gallery[:debug_limit]
        print(f"[DEBUG] 截断: query={len(dataset.query)}, gallery={len(dataset.gallery)}")

    # 构建模型
    print("[Init] 构建模型...")
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)

    if cfg.TEST.WEIGHT:
        print(f"[Init] 加载权重: {cfg.TEST.WEIGHT}")
        # 用 map_location='cpu' 避免在 CUDA 不可用时加载报错
        param_dict = torch.load(cfg.TEST.WEIGHT, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            key = i.replace('module.', '')
            if key in model.state_dict():
                model.state_dict()[key].copy_(param_dict[i])
        print(f"[Init] 权重加载完成")
    else:
        print("[Init] Warning: No TEST.WEIGHT specified, using random weights.")
        print("       The matching will be random and meaningless - only for pipeline testing.")

    model.to(device)
    model.eval()

    # 提取所有特征
    print("[特征提取] 开始提取 query 和 gallery 特征...")
    t0 = time.time()

    # 直接用 BallShow 的 query/gallery 列表，配合 PIL 读取 + transform
    img_size = cfg.INPUT.SIZE_TEST
    val_transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    query_data = []   # [(img_path, pid, camid, feat_tensor)]
    gallery_data = [] # [(img_path, pid, camid, feat_tensor)]

    # ===== 提取 query 特征（分批加速） =====
    print(f"  提取 {len(dataset.query)} 张 query 特征（分批）...")
    batch_size_q = 64
    with torch.no_grad():
        for i in range(0, len(dataset.query), batch_size_q):
            batch = dataset.query[i:i+batch_size_q]
            imgs, cam_labels, view_labels = [], [], []
            for img_path, pid, camid, trackid in batch:
                imgs.append(val_transforms(_read_image_pil(img_path)))
                cam_labels.append(camid)
                view_labels.append(trackid)

            try:
                imgs_t = torch.stack(imgs).to(device)
                cam_labels_t = torch.tensor(cam_labels).to(device)
                view_labels_t = torch.tensor(view_labels).to(device)
                feats = model(imgs_t, cam_label=cam_labels_t, view_label=view_labels_t).cpu()
            except RuntimeError as e:
                print(f"\n    [ERROR] query batch {i}-{i+len(batch)} 维度错误: {e}")
                print(f"    batch size: {len(batch)}, feats type: {type(feats)}")
                print(f"    cam_labels_t shape: {torch.tensor(cam_labels).shape}")
                print(f"    view_labels_t shape: {torch.tensor(view_labels).shape}")
                raise

            for j, (img_path, pid, camid, trackid) in enumerate(batch):
                query_data.append((img_path, pid, camid, feats[j]))

            done_q = min(i+batch_size_q, len(dataset.query))
            print(f"    query 进度: {done_q}/{len(dataset.query)}")

    # ===== 提取 gallery 特征（分批） =====
    print(f"  提取 {len(dataset.gallery)} 张 gallery 特征（分批）...")
    batch_size_g = 64
    with torch.no_grad():
        for i in range(0, len(dataset.gallery), batch_size_g):
            batch = dataset.gallery[i:i+batch_size_g]
            imgs, cam_labels, view_labels = [], [], []
            for img_path, pid, camid, trackid in batch:
                imgs.append(val_transforms(_read_image_pil(img_path)))
                cam_labels.append(camid)
                view_labels.append(trackid)

            try:
                imgs_t = torch.stack(imgs).to(device)
                cam_labels_t = torch.tensor(cam_labels).to(device)
                view_labels_t = torch.tensor(view_labels).to(device)
                feats = model(imgs_t, cam_label=cam_labels_t, view_label=view_labels_t).cpu()
            except RuntimeError as e:
                print(f"\n    [ERROR] gallery batch {i}-{i+len(batch)} 维度错误: {e}")
                print(f"    batch size: {len(batch)}, cam_labels: {cam_labels[:5]}, view_labels: {view_labels[:5]}")
                print(f"    imgs shape after stack: {[t.shape for t in imgs[:3]]}")
                print(f"    cam_labels_t shape: {torch.tensor(cam_labels).shape}")
                print(f"    view_labels_t shape: {torch.tensor(view_labels).shape}")
                raise

            for j, (img_path, pid, camid, trackid) in enumerate(batch):
                gallery_data.append((img_path, pid, camid, feats[j]))

            done_g = min(i+batch_size_g, len(dataset.gallery))
            print(f"    gallery 进度: {done_g}/{len(dataset.gallery)}")

    t1 = time.time()
    print(f"[特征提取] 完成！耗时 {t1-t0:.1f}s")

    # ===== 先对全部 query 做一次预检索，找出正确和错误的样本 =====
    print(f"\n[预检索] 对 {len(query_data)} 张 query 做预检索...")
    correct_indices = []
    wrong_indices = []
    for idx, (q_path, q_pid, q_camid, q_feat) in enumerate(query_data):
        res = search_top_k(q_feat, gallery_data, top_k=1)
        if res and res[0][1] == q_pid:
            correct_indices.append(idx)
        else:
            wrong_indices.append(idx)
    print(f"[预检索] Top-1 正确: {len(correct_indices)}, 错误: {len(wrong_indices)}")

    # ===== 抽样策略：全部随机抽取 =====
    random.seed(42)
    all_rounds = []

    used_indices = set()
    for _ in range(num_rounds):
        while True:
            q_idx = random.randint(0, len(query_data) - 1)
            if q_idx not in used_indices:
                used_indices.add(q_idx)
                break
        q_path, q_pid, q_camid, q_feat = query_data[q_idx]
        results = search_top_k(q_feat, gallery_data, top_k=top_k)
        top1_correct = (results[0][1] == q_pid) if results else False

        # 找正确答案在 Top-K 里的排名
        correct_rank = None
        for rk, r in enumerate(results):
            if r[1] == q_pid:
                correct_rank = rk + 1
                break

        round_result = {
            'query_path': q_path, 'query_pid': q_pid, 'query_camid': q_camid,
            'top1_correct': top1_correct, 'top_k': top_k,
            'strategy': 'success_case' if top1_correct else 'failure_case',
            'results': [{'path': r[0], 'pid': r[1], 'camid': r[2], 'distance': r[3]} for r in results]
        }
        all_rounds.append(round_result)

        status = "Correct" if top1_correct else "Wrong"
        rank_info = f" | 正确答案在第 {correct_rank} 位" if not top1_correct and correct_rank else ""
        print(f"  第 {len(all_rounds):2d} 轮 | Query PID={q_pid:>3d} | Top-1 PID={results[0][1]:>3d} (dist={results[0][3]:.4f}) | {status}{rank_info}")


    # 统计
    correct_count = sum(1 for r in all_rounds if r['top1_correct'])
    print(f"\n{'='*50}")
    print(f"  Top-1 命中率: {correct_count}/{len(all_rounds)} = {correct_count/len(all_rounds)*100:.1f}%")
    print(f"{'='*50}")

    # 复制图片并生成 HTML 报告
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_dir = os.path.join(cfg.OUTPUT_DIR, "visual_test")
    os.makedirs(report_dir, exist_ok=True)

    # 复制 query 和检索结果的图片到 report_dir/images/
    img_dir = os.path.join(report_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    path_map = {}  # 原路径 -> 相对路径
    for rd in all_rounds:
        # query 图片
        q_src = rd['query_path']
        q_dst_name = f"query_r{all_rounds.index(rd)+1}_{os.path.basename(q_src)}"
        q_dst = os.path.join(img_dir, q_dst_name)
        if q_src not in path_map:
            shutil.copy2(q_src, q_dst)
            path_map[q_src] = f"images/{q_dst_name}"
        rd['query_path'] = path_map[q_src]

        # 结果图片
        for r in rd['results']:
            r_src = r['path']
            if r_src not in path_map:
                r_dst_name = f"gallery_{os.path.basename(r_src)}"
                r_dst = os.path.join(img_dir, r_dst_name)
                # 避免文件名冲突
                counter = 1
                while os.path.exists(r_dst):
                    name, ext = os.path.splitext(r_dst_name)
                    r_dst = os.path.join(img_dir, f"{name}_{counter}{ext}")
                    counter += 1
                shutil.copy2(r_src, r_dst)
                path_map[r_src] = f"images/{os.path.basename(r_dst)}"
            r['path'] = path_map[r_src]

    # 生成 HTML
    html_path = os.path.join(report_dir, "visual_test_report.html")

    # 重写 _get_img_src 以使用 path_map 中的相对路径
    model_info = f"ViT-Base + JPM + SIE (Stride {cfg.MODEL.STRIDE_SIZE})"
    generate_html_report_with_paths(all_rounds, html_path, model_info, timestamp)

    print(f"\n[完成] 报告已保存至: {html_path}")
    print(f"       图片目录: {img_dir}")
    return all_rounds


def generate_html_report_with_paths(all_rounds, output_path, model_info, timestamp):
    """生成 HTML 报告（使用已经替换好的相对路径）"""

    total = len(all_rounds)
    correct = sum(1 for r in all_rounds if r['top1_correct'])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>TransReID 可视化验证报告</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f0f2f5; color: #333; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
              color: white; padding: 30px 40px; text-align: center; }}
  .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
  .header .meta {{ font-size: 14px; opacity: 0.8; }}
  .stats {{ display: flex; justify-content: center; gap: 40px; padding: 24px 20px;
            background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 24px auto;
            max-width: 650px; border-radius: 12px; }}
  .stat-item {{ text-align: center; }}
  .stat-item .num {{ font-size: 36px; font-weight: bold; }}
  .stat-item .num.green {{ color: #27ae60; }}
  .stat-item .num.red {{ color: #e74c3c; }}
  .stat-item .label {{ font-size: 13px; color: #888; margin-top: 6px; }}
  .round {{ background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            margin: 20px auto; max-width: 1150px; padding: 28px; }}
  .round-title {{ font-size: 18px; font-weight: bold; margin-bottom: 18px;
                 padding-bottom: 12px; border-bottom: 2px solid #f0f0f0;
                 display: flex; align-items: center; gap: 12px; }}
  .badge {{ display: inline-block; padding: 3px 12px; border-radius: 14px;
            font-size: 12px; font-weight: 600; }}
  .badge-correct {{ background: #d4edda; color: #155724; }}
  .badge-wrong {{ background: #f8d7da; color: #721c24; }}
  .query-section {{ margin-bottom: 18px; }}
  .section-label {{ font-size: 14px; font-weight: 600; color: #555; margin-bottom: 8px; }}
  .query-info {{ font-size: 13px; color: #777; margin-top: 4px; line-height: 1.6; }}
  .img-row {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start; }}
  .query-img-box {{ border: 3px solid #3498db; border-radius: 8px; overflow: hidden; flex-shrink: 0;
                   box-shadow: 0 2px 8px rgba(52,152,219,0.3); }}
  .query-img-box img {{ display: block; height: 180px; width: auto; }}
  .results-grid {{ display: flex; gap: 14px; flex-wrap: wrap; }}
  .result-card {{ border: 2px solid #e8e8e8; border-radius: 8px; overflow: hidden;
                  text-align: center; width: 150px; flex-shrink: 0;
                  transition: transform 0.2s, box-shadow 0.2s; }}
  .result-card:hover {{ transform: translateY(-4px); box-shadow: 0 6px 16px rgba(0,0,0,0.12); }}
  .result-card.rank1-correct {{ border-color: #27ae60; box-shadow: 0 0 0 2px rgba(39,174,96,0.2); }}
  .result-card.rank1-wrong {{ border-color: #e74c3c; box-shadow: 0 0 0 2px rgba(231,76,60,0.2); }}
  .result-card img {{ display: block; height: 150px; width: auto; margin: 0 auto; }}
  .result-card .info {{ padding: 8px 6px; font-size: 11px; }}
  .result-card .rank {{ font-weight: bold; font-size: 14px; color: #333; }}
  .result-card .dist {{ color: #888; }}
  .result-card .match {{ font-size: 12px; margin-top: 3px; font-weight: 500; }}
  .match-yes {{ color: #27ae60; }}
  .match-no {{ color: #e74c3c; }}
  .footer {{ text-align: center; padding: 24px; color: #bbb; font-size: 12px; }}
  .separator {{ height: 1px; background: #e8e8e8; margin: 14px 0; }}
</style>
</head>
<body>

<div class="header">
  <h1>TransReID 篮球持球人 ReID - 可视化验证报告</h1>
  <div class="meta">{model_info} | 生成时间: {timestamp} | 共 {total} 轮测试</div>
</div>

<div class="stats">
  <div class="stat-item">
    <div class="num">{total}</div>
    <div class="label">总测试轮数</div>
  </div>
  <div class="stat-item">
    <div class="num green">{correct}</div>
    <div class="label">Top-1 命中</div>
  </div>
  <div class="stat-item">
    <div class="num {'green' if correct/total >= 0.8 else 'red'}">{correct/total*100:.1f}%</div>
    <div class="label">Top-1 命中率</div>
  </div>
</div>

"""

    for i, rd in enumerate(all_rounds):
        status_class = "badge-correct" if rd['top1_correct'] else "badge-wrong"
        status_text = "Top-1 命中" if rd['top1_correct'] else "Top-1 未命中"
        q_path = rd['query_path']
        strategy = rd.get('strategy', 'random')
        if strategy == 'failure_case':
            strategy_label = "❌ 失败案例"
            strategy_badge_class = "badge-wrong"
        elif strategy == 'success_case':
            strategy_label = "✅ 成功案例"
            strategy_badge_class = "badge-correct"
        else:
            strategy_label = "随机抽取"
            strategy_badge_class = ""

        html += f"""
<div class="round">
  <div class="round-title">
    Round {i+1}
    <span class="badge {status_class}">{status_text}</span>
    <span class="badge {strategy_badge_class}" style="background:#fff3cd;color:#856404;">{strategy_label}</span>
  </div>

  <div class="query-section">
    <div class="section-label">Query (PID: {rd['query_pid']}, Camera: {rd['query_camid']})</div>
    <div class="img-row">
      <div class="query-img-box">
        <img src="{q_path}" />
      </div>
      <div style="padding-top:8px;">
        <div class="query-info">文件: {os.path.basename(q_path)}</div>
        <div class="query-info">真实 PID: {rd['query_pid']} | Camera: {rd['query_camid']}</div>
      </div>
    </div>
  </div>

  <div class="section-label" style="margin-top:14px;">Top-{rd['top_k']} Retrieval Results</div>
  <div class="separator"></div>
  <div class="results-grid" style="margin-top:12px;">
"""
        for j, r in enumerate(rd['results']):
            is_correct = (r['pid'] == rd['query_pid'])
            is_junk = (r['pid'] == rd['query_pid'] and r['camid'] == rd['query_camid'])

            if j == 0:
                card_class = "rank1-correct" if is_correct else "rank1-wrong"
            else:
                card_class = ""
            match_class = "match-yes" if is_correct else "match-no"
            match_text = "Match" if is_correct else "Mismatch"
            if is_junk:
                match_text += " (same cam)"

            html += f"""
    <div class="result-card {card_class}">
      <img src="{r['path']}" />
      <div class="info">
        <div class="rank">Rank-{j+1}</div>
        <div class="dist">dist: {r['distance']:.4f}</div>
        <div class="match {match_class}">{match_text}</div>
        <div style="color:#aaa;font-size:10px;margin-top:2px;">PID:{r['pid']} Cam:{r['camid']}</div>
      </div>
    </div>
"""

        html += """
  </div>
</div>
"""

    html += f"""
<div class="footer">
  TransReID Visual Verification Report | Generated by visual_test.py
</div>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == "__main__":
    main()
