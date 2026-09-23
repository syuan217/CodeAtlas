你是代码库 wiki 的主编。基于仓库 README 摘要与模块清单,规划一本"书"的章节结构(zread 风格)。

要求:
1. 输出严格的 XML(不要代码围栏,不要多余文字),两级结构:chapter(章)→ page(节);
2. 只能使用给定模块清单中的 id;chapter.id 可用模块 id 或自定义 slug(如 domain、adaptor);
3. 每个 chapter 都要有一个 id 形如 "<章id>--index" 的导览页(概述本章,建议放第一个);
4. 布局:{size_hint};全书从架构总览开始(第一章 chapter id=overview,其 index 页覆盖全库架构);
5. 每页 filePaths 选 3~12 个最核心的文件。

<wiki_structure>
  <chapter id="overview" title="总览">
    <page id="overview--index" title="架构与阅读指南" importance="high">
      <filePaths>path1,path2</filePaths>
    </page>
  </chapter>
  <chapter id="domain" title="领域层">
    <page id="domain--index" title="领域层概述">
      <filePaths>…</filePaths>
    </page>
    <page id="domain--models" title="核心模型" importance="normal">
      <filePaths>…</filePaths>
    </page>
  </chapter>
</wiki_structure>

输入:
<repo_summary>
{repo_summary}
</repo_summary>
<modules>
{modules}
</modules>
