import pandas as pd
import streamlit as st
import numpy as np
from io import BytesIO
from datetime import datetime, time

# ==================== 数据缓存 ====================
@st.cache_data
def load_data(uploaded_file):
    """缓存数据读取"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

@st.cache_data
def process_data(df):
    """缓存数据处理结果"""
    
    # 确保时间格式正确
    date_cols = ['订单提交时间', '合作开始时间', '合作到期时间']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 1. 分离基础版和加油包
    base_df = df[df['附属产品'] == '按量计费基础版'].copy()
    addon_df = df[df['附属产品'] == '按量计费加油包(1000份)'].copy()
    
    if len(base_df) == 0:
        return None
    
    # 2. 按客户ID + 合作到期时间 聚合加油包
    if len(addon_df) > 0:
        addon_agg = addon_df.groupby(['客户ID', '合作到期时间']).agg({
            '签约金额': 'sum',
            '购买份数': 'sum',
            '订单号': 'count'
        }).rename(columns={
            '签约金额': '加油包金额',
            '购买份数': '加油包份数',
            '订单号': '加油包订单数'
        }).reset_index()
    else:
        addon_agg = pd.DataFrame(columns=['客户ID', '合作到期时间', '加油包金额', '加油包份数', '加油包订单数'])
    
    # 3. 合并基础版与加油包
    result = base_df.merge(
        addon_agg,
        on=['客户ID', '合作到期时间'],
        how='left'
    )
    
    result['加油包金额'] = result['加油包金额'].fillna(0)
    result['加油包份数'] = result['加油包份数'].fillna(0).astype(int)
    result['加油包订单数'] = result['加油包订单数'].fillna(0).astype(int)
    
    # 4. 计算汇总字段
    result['订单总金额'] = result['签约金额'] + result['加油包金额']
    result['订单总份数'] = result['购买份数'] + result['加油包份数']
    
    result = result.rename(columns={
        '签约金额': '基础版签约金额',
        '购买份数': '基础版份数',
        '合作到期时间': '合作结束时间'
    })
    
    # 5. 计算续费信息
    result = calculate_renewal(result)
    
    # 6. 添加档位
    result = add_tier_interval(result)
    result = add_tier_nearest(result)
    
    return result

def add_expiry_and_consumption(df, current_time):
    """添加到期和消耗相关字段，使用传入的 current_time"""
    
    # 1. 当前订单是否已到期
    def is_expired(end_time):
        if pd.isna(end_time):
            return '未知'
        return '是' if end_time < current_time else '否'
    
    df['当前订单是否已到期'] = df['合作结束时间'].apply(is_expired)
    
    # 2. 当前客户最后一个订单是否已到期
    # 先找到每个客户最后一个订单的结束时间
    last_order = df.groupby('客户ID')['合作结束时间'].max().reset_index()
    last_order['客户最后一个订单是否已到期'] = last_order['合作结束时间'].apply(is_expired)
    # 合并回原df
    df = df.merge(last_order[['客户ID', '客户最后一个订单是否已到期']], on='客户ID', how='left')
    
    # 3. 消耗天数
    def calculate_consumption(row):
        if pd.isna(row['合作开始时间']) or pd.isna(row['合作结束时间']):
            return None
        if row['当前订单是否已到期'] == '是':
            return 365
        else:
            return (current_time - row['合作开始时间']).days
    
    df['消耗天数'] = df.apply(calculate_consumption, axis=1)
    
    return df

def calculate_renewal(df):
    """计算续费相关字段"""
    
    df = df.sort_values(['客户ID', '合作结束时间']).reset_index(drop=True)
    
    renewal_results = []
    
    for idx, row in df.iterrows():
        customer_id = row['客户ID']
        current_end = row['合作结束时间']
        current_base_amount = row['基础版签约金额']
        
        future_orders = df[
            (df['客户ID'] == customer_id) & 
            (df['合作结束时间'] > current_end)
        ].sort_values('合作结束时间')
        
        if len(future_orders) > 0:
            next_order = future_orders.iloc[0]
            is_renewed = '是'
            
            if pd.notna(next_order['订单提交时间']) and pd.notna(current_end):
                renewal_interval = (next_order['订单提交时间'] - current_end).days
            else:
                renewal_interval = None
                
            renewal_total = next_order['订单总金额']
            renewal_base = next_order['基础版签约金额']
            renewal_submit_time = next_order['订单提交时间']
            renewal_end_time = next_order['合作结束时间']
            
            if renewal_base > current_base_amount:
                change_type = '升档'
            elif renewal_base < current_base_amount:
                change_type = '降档'
            else:
                change_type = '不变'
        else:
            is_renewed = '否'
            renewal_interval = None
            renewal_total = None
            renewal_base = None
            renewal_submit_time = None
            renewal_end_time = None
            change_type = '流失'
        
        renewal_results.append({
            '是否续费': is_renewed,
            '续费间隔时间(天)': renewal_interval,
            '续费订单提交时间': renewal_submit_time,
            '续费订单结束时间': renewal_end_time,
            '续费总金额': renewal_total,
            '续费订单基础版金额': renewal_base,
            '档位变化': change_type
        })
    
    renewal_df = pd.DataFrame(renewal_results)
    return pd.concat([df.reset_index(drop=True), renewal_df], axis=1)


def add_tier_interval(df, amount_col='基础版签约金额'):
    """方式1：区间划分"""
    bins = [0, 1000, 2000, 3500, 5000, float('inf')]
    labels = ['0-1000', '1000-2000', '2000-3500', '3500-5000', '5000以上']
    df['金额档位(区间)'] = pd.cut(df[amount_col], bins=bins, labels=labels, right=True, include_lowest=True)
    return df


def add_tier_nearest(df, amount_col='基础版签约金额'):
    """方式2：就近匹配"""
    tiers = [1000, 2000, 3500, 5000]
    
    def find_nearest(value):
        if pd.isna(value) or value == 0:
            return 0
        nearest = min(tiers, key=lambda x: abs(x - value))
        return nearest
    
    df['金额档位(就近)'] = df[amount_col].apply(find_nearest)
    return df


def generate_stats(df, group_col):
    """生成分组统计"""
    stats = df.groupby(group_col, observed=True).agg(
        订单数=('客户ID', 'count'),
        客户数=('客户ID', 'nunique'),
        订单总金额=('订单总金额', 'sum'),
        基础版金额合计=('基础版签约金额', 'sum'),
        加油包金额合计=('加油包金额', 'sum'),
        续费数=('是否续费', lambda x: (x == '是').sum()),
        升档数=('档位变化', lambda x: (x == '升档').sum()),
        降档数=('档位变化', lambda x: (x == '降档').sum()),
        不变数=('档位变化', lambda x: (x == '不变').sum()),
        流失数=('档位变化', lambda x: (x == '流失').sum())
    ).reset_index()
    
    stats['续费率'] = (stats['续费数'] / stats['订单数'] * 100).round(2)
    stats['流失率'] = (stats['流失数'] / stats['订单数'] * 100).round(2)
    
    return stats


# ==================== Streamlit 界面 ====================

def main():
    st.set_page_config(
        page_title="订单续费分析工具", 
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 订单续费分析工具")
    st.caption("按量计费基础版 & 加油包订单分析")
    
    # ========== 侧边栏：文件上传 ==========
    st.sidebar.header("📁 数据上传")
    uploaded_file = st.sidebar.file_uploader(
        "上传CSV或Excel文件", 
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success(f"✅ 已加载 {len(df)} 条原始记录")
            
            with st.sidebar.expander("📋 原始数据概览"):
                base_count = len(df[df['附属产品'] == '按量计费基础版'])
                addon_count = len(df[df['附属产品'] == '按量计费加油包(1000份)'])
                st.write(f"基础版订单: {base_count} 条")
                st.write(f"加油包订单: {addon_count} 条")
                st.write(f"客户数: {df['客户ID'].nunique()}")
                
        except Exception as e:
            st.error(f"❌ 文件读取失败: {str(e)}")
            return
        
        # 处理数据（有缓存）
        with st.spinner("🔄 正在处理数据..."):
            result = process_data(df)
        
        if result is None:
            st.error("❌ 数据中没有找到「按量计费基础版」订单")
            return
            
        st.sidebar.success(f"✅ 处理完成，共 {len(result)} 条基础版订单")
        
        # ========== 侧边栏：筛选器（使用表单避免实时刷新） ==========
        st.sidebar.header("🔍 筛选条件")
        st.sidebar.caption("设置好条件后点击「应用筛选」")
        
        with st.sidebar.form(key='filter_form'):
            # 时间筛选
            date_dimension = st.selectbox(
                "时间筛选维度",
                ["订单提交时间", "合作开始时间", "合作结束时间"]
            )
            
            valid_dates = result[date_dimension].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                
                col1, col2 = st.columns(2)
                with col1:
                    start_datetime = st.datetime_input(
                        "开始日期时间 (可选: >= 此时间)",
                        value=None,
                        min_value=min_date,
                        max_value=max_date
                    )
                with col2:
                    end_datetime = st.datetime_input(
                        "结束日期时间 (可选: <= 此时间)",
                        value=None,
                        min_value=min_date,
                        max_value=max_date
                    )
            else:
                start_datetime = None
                end_datetime = None
            
            st.divider()
            
            # 新增：模拟当前时间设置
            simulated_current_time = st.datetime_input(
                "模拟当前时间 (用于到期计算)",
                value=datetime.now(),
                help="设置一个自定义时间，用于计算到期和消耗天数。默认使用系统当前时间。"
            )
            
            st.divider()
            
            # 新增：基础版签约金额筛选
            if '基础版签约金额' in result.columns:
                amount_col = '基础版签约金额'
                min_amount = float(result[amount_col].min())
                max_amount = float(result[amount_col].max())
                
                st.subheader("💰 基础版签约金额筛选")
                amount_range = st.slider(
                    "金额范围",
                    min_value=min_amount,
                    max_value=max_amount,
                    value=(min_amount, max_amount)
                )
                
                unique_amounts = sorted(result[amount_col].unique())
                amount_filter = st.multiselect(
                    "特定金额值",
                    options=unique_amounts,
                    default=[]
                )
            else:
                amount_range = None
                amount_filter = None
            
            st.divider()
            
            # 续费状态筛选
            renewal_filter = st.multiselect(
                "续费状态",
                options=['是', '否'],
                default=['是', '否']
            )
            
            # 档位变化筛选
            change_filter = st.multiselect(
                "档位变化",
                options=['升档', '降档', '不变', '流失'],
                default=['升档', '降档', '不变', '流失']
            )
            
            # 客户类型筛选
            if '客户类型' in result.columns:
                customer_types = result['客户类型'].dropna().unique().tolist()
                type_filter = st.multiselect(
                    "客户类型",
                    options=customer_types,
                    default=customer_types
                )
            else:
                type_filter = None
            
            st.divider()
            
            # 🔘 提交按钮
            submit_button = st.form_submit_button(
                label="🔍 应用筛选",
                use_container_width=True,
                type="primary"
            )
        
        # ========== 应用筛选 ==========
        filtered = result.copy()
        
        # 时间筛选：支持开区间
        if start_datetime:
            filtered = filtered[filtered[date_dimension] >= start_datetime]
        if end_datetime:
            filtered = filtered[filtered[date_dimension] <= end_datetime]
        
        # 金额筛选
        if amount_range and '基础版签约金额' in filtered.columns:
            filtered = filtered[
                (filtered['基础版签约金额'] >= amount_range[0]) &
                (filtered['基础版签约金额'] <= amount_range[1])
            ]
        if amount_filter and '基础版签约金额' in filtered.columns:
            filtered = filtered[filtered['基础版签约金额'].isin(amount_filter)]
        
        if renewal_filter:
            filtered = filtered[filtered['是否续费'].isin(renewal_filter)]
        
        if change_filter:
            filtered = filtered[filtered['档位变化'].isin(change_filter)]
        
        if type_filter and '客户类型' in filtered.columns:
            filtered = filtered[filtered['客户类型'].isin(type_filter)]
        
        # 新增：基于模拟当前时间计算到期和消耗字段
        filtered = add_expiry_and_consumption(filtered, simulated_current_time)
        
        # ========== 主区域：数据展示 ==========
        
        # 顶部指标卡片
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📋 订单数", f"{len(filtered):,}")
        with col2:
            st.metric("👥 客户数", f"{filtered['客户ID'].nunique():,}")
        with col3:
            st.metric("💰 总金额", f"¥{filtered['订单总金额'].sum():,.0f}")
        with col4:
            renewal_rate = (filtered['是否续费'] == '是').sum() / len(filtered) * 100 if len(filtered) > 0 else 0
            st.metric("🔄 续费率", f"{renewal_rate:.1f}%")
        with col5:
            churn_rate = (filtered['档位变化'] == '流失').sum() / len(filtered) * 100 if len(filtered) > 0 else 0
            st.metric("📉 流失率", f"{churn_rate:.1f}%")
        
        st.divider()
        
        # Tab页
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 明细数据", 
            "📈 区间档位统计", 
            "📊 就近档位统计",
            "🏢 客户类型统计"
        ])
        
        with tab1:
            st.subheader(f"筛选后数据：{len(filtered)} 条")
            
            display_cols = [
                '客户ID', '客户名称', '客户类型', '客户学段',
                '订单提交时间', '合作开始时间', '合作结束时间',
                '基础版签约金额', '基础版份数', 
                '加油包金额', '加油包份数',
                '订单总金额', '订单总份数',
                '是否续费', '续费间隔时间(天)', 
                '续费订单提交时间', '续费总金额', '续费订单基础版金额', 
                '档位变化',
                '金额档位(区间)', '金额档位(就近)',
                '签约类型',
                # 新增字段
                '当前订单是否已到期', '客户最后一个订单是否已到期', '消耗天数'
            ]
            display_cols = [c for c in display_cols if c in filtered.columns]
            
            st.dataframe(
                filtered[display_cols], 
                use_container_width=True,
                height=500
            )
        
        with tab2:
            st.subheader("📈 按金额区间统计")
            stats1 = generate_stats(filtered, '金额档位(区间)')
            
            display_stats1 = stats1.copy()
            display_stats1['订单总金额'] = display_stats1['订单总金额'].apply(lambda x: f"¥{x:,.0f}")
            display_stats1['基础版金额合计'] = display_stats1['基础版金额合计'].apply(lambda x: f"¥{x:,.0f}")
            display_stats1['加油包金额合计'] = display_stats1['加油包金额合计'].apply(lambda x: f"¥{x:,.0f}")
            display_stats1['续费率'] = display_stats1['续费率'].apply(lambda x: f"{x}%")
            display_stats1['流失率'] = display_stats1['流失率'].apply(lambda x: f"{x}%")
            
            st.dataframe(display_stats1, use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**订单数分布**")
                chart_data1 = filtered.groupby('金额档位(区间)', observed=True)['客户ID'].count()
                st.bar_chart(chart_data1)
            with col2:
                st.write("**续费率分布**")
                st.bar_chart(stats1.set_index('金额档位(区间)')['续费率'])
        
        with tab3:
            st.subheader("📊 按就近档位统计 (1000/2000/3500/5000)")
            stats2 = generate_stats(filtered, '金额档位(就近)')
            
            display_stats2 = stats2.copy()
            display_stats2['订单总金额'] = display_stats2['订单总金额'].apply(lambda x: f"¥{x:,.0f}")
            display_stats2['基础版金额合计'] = display_stats2['基础版金额合计'].apply(lambda x: f"¥{x:,.0f}")
            display_stats2['加油包金额合计'] = display_stats2['加油包金额合计'].apply(lambda x: f"¥{x:,.0f}")
            display_stats2['续费率'] = display_stats2['续费率'].apply(lambda x: f"{x}%")
            display_stats2['流失率'] = display_stats2['流失率'].apply(lambda x: f"{x}%")
            
            st.dataframe(display_stats2, use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**订单数分布**")
                chart_data2 = filtered.groupby('金额档位(就近)', observed=True)['客户ID'].count()
                st.bar_chart(chart_data2)
            with col2:
                st.write("**档位变化分布**")
                change_counts = filtered['档位变化'].value_counts()
                st.bar_chart(change_counts)
        
        with tab4:
            st.subheader("🏢 按客户类型统计")
            if '客户类型' in filtered.columns:
                stats3 = generate_stats(filtered, '客户类型')
                
                display_stats3 = stats3.copy()
                display_stats3['订单总金额'] = display_stats3['订单总金额'].apply(lambda x: f"¥{x:,.0f}")
                display_stats3['续费率'] = display_stats3['续费率'].apply(lambda x: f"{x}%")
                display_stats3['流失率'] = display_stats3['流失率'].apply(lambda x: f"{x}%")
                
                st.dataframe(display_stats3, use_container_width=True, hide_index=True)
            else:
                st.info("数据中没有客户类型字段")
        
        # ========== 下载按钮 ==========
        st.sidebar.header("📥 导出数据")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_cols = [c for c in display_cols if c in filtered.columns]
            filtered[export_cols].to_excel(writer, sheet_name='明细数据', index=False)
            stats1.to_excel(writer, sheet_name='区间档位统计', index=False)
            stats2.to_excel(writer, sheet_name='就近档位统计', index=False)
            if '客户类型' in filtered.columns:
                stats3.to_excel(writer, sheet_name='客户类型统计', index=False)
        
        st.sidebar.download_button(
            label="📥 下载Excel报表",
            data=output.getvalue(),
            file_name=f"订单续费分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    else:
        st.info("👈 请在左侧上传CSV或Excel文件开始分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 数据格式要求")
            st.markdown("""
            **必需字段：**
            | 字段名 | 说明 |
            |--------|------|
            | 附属产品 | `按量计费基础版` 或 `按量计费加油包(1000份)` |
            | 订单提交时间 | 日期时间格式 |
            | 客户ID | 客户唯一标识 |
            | 签约金额 | 数值类型 |
            | 合作开始时间 | 日期格式 |
            | 合作到期时间 | 日期格式 |
            | 购买份数 | 数值类型 |
            """)
        
        with col2:
            st.subheader("🔧 功能说明")
            st.markdown("""
            **数据处理：**
            - 同客户ID + 同合作到期时间的基础版和加油包合并
            - 自动计算订单总金额、总份数
            
            **续费判断：**
            - 合作结束后是否有新的基础版订单
            - 续费间隔 = 续费订单提交时间 - 当前订单结束时间
            
            **档位分析：**
            - 升档/降档/不变/流失
            """)

if __name__ == "__main__":
    main()