import matplotlib.pyplot as plt
import os


class FormulaImageGenerator:

    def __init__(self, output_dir="latex_img", fontsize=22, dpi=300):
        self.output_dir = output_dir
        self.fontsize = fontsize
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    def render_formula(self, name, formula, explanations):
        """渲染公式和解释为一张图"""
        filename = os.path.join(self.output_dir, f"{name}.png")

        fig = plt.figure(figsize=(8, 4), dpi=self.dpi)
        plt.axis("off")

        # 主公式区域
        title_text = f"{name}:\n${formula}$"

        # 解释列表
        explain_lines = [f"- ${k}$：{v}" for k, v in explanations.items()]
        explain_text = "\n".join(explain_lines)

        plt.text(0.5, 0.75, title_text,
                 fontsize=self.fontsize, ha='center', va='center')

        plt.text(0.05, 0.10, explain_text,
                 fontsize=self.fontsize * 0.8, ha='left', va='bottom')

        plt.savefig(filename, bbox_inches='tight', pad_inches=0.3)
        plt.close()
        print(f"生成: {filename}")

    def generate_batch(self, formulas_dict):
        """批量生成所有公式"""
        for name, item in formulas_dict.items():
            formula = item["formula"]
            explanations = item["explain"]
            self.render_formula(name, formula, explanations)
