#!/bin/bash
# 下载真实工业检测数据集

echo "🏭 下载真实工业检测数据集..."

# 选项1: 使用公开的工业检测数据集
echo "1️⃣ 推荐数据集："
echo "   - Open Images V7 (有工业物体类别)"
echo "   - PASCAL VOC + Industrial objects"  
echo "   - Roboflow Industrial Dataset"
echo "   - Custom Industrial Parts Dataset"

echo ""
echo "2️⃣ 快速开始 - 使用小规模真实数据集："

# 创建一个最小化但真实的工业数据集模板
mkdir -p /home/minsea01/datasets/real_industrial_10cls/{train,val,test}/{images,labels}

cat > /home/minsea01/datasets/real_industrial_10cls/data.yaml << 'EOF'
# 真实工业10类检测数据集
path: /home/minsea01/datasets/real_industrial_10cls
train: train/images
val: val/images  
test: test/images

nc: 10
names:
  0: screw          # 螺丝钉
  1: bolt           # 螺栓  
  2: nut            # 螺母
  3: washer         # 垫圈
  4: gear           # 齿轮
  5: bearing        # 轴承
  6: valve          # 阀门
  7: connector      # 连接器
  8: circuit_board  # 电路板
  9: defect         # 缺陷检测
EOF

echo "✅ 数据集模板创建完成: /home/minsea01/datasets/real_industrial_10cls/"
echo ""
echo "3️⃣ 下一步："
echo "   你需要收集真实的工业图像和标注，或者："
echo "   - 购买商业工业数据集"
echo "   - 使用开源工业数据集"
echo "   - 自己拍摄标注工业场景"
echo ""
echo "⚠️ 不要再使用合成/映射的数据集了！"