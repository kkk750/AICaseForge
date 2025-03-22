<template>
  <div class="workbench">
    <div class="container-fluid py-4">
      <div class="row">
        <div class="col-12">
          <ul class="nav nav-tabs" role="tablist">
            <li class="nav-item" role="presentation">
              <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab1" type="button" role="tab">
                <h5 class="mb-1">大模型生成测试用例</h5>
                <p class="small mb-0">基于智能搜索和大模型的知识库，自动化生成测试用例</p>
              </button>
            </li>
            <li class="nav-item" role="presentation">
              <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab2" type="button" role="tab">
                <h5 class="mb-1">上传文件至向量数据库</h5>
                <p class="small mb-0">解析产品设计文档，将数据以向量形式存储在数据库中，作为大模型的外挂知识库</p>
              </button>
            </li>
            <li class="nav-item" role="presentation">
              <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab3" type="button" role="tab">
                <h5 class="mb-1">向量数据表空间操作</h5>
                <p class="small mb-0">管理您的向量数据表，包含了增删改查等多种操作</p>
              </button>
            </li>
          </ul>

          <div class="tab-content mt-3">
            <div class="tab-pane fade show active" id="tab1" role="tabpanel">
              <!-- 测试用例生成面板 -->
              <div class="card">
                <div class="card-body">
                  <h4>测试用例生成</h4>
                  <!-- 这里添加测试用例生成的具体内容 -->
                </div>
              </div>
            </div>

            <div class="tab-pane fade" id="tab2" role="tabpanel">
              <!-- 文件上传面板 -->
              <div class="card">
                <div class="card-body">
                  <h4>文件上传</h4>
                  <!-- 这里添加文件上传的具体内容 -->
                </div>
              </div>
            </div>

            <div class="tab-pane fade" id="tab3" role="tabpanel">
              <!-- 数据表操作面板 -->
              <div class="card">
                <div class="card-body">
                  <h4>数据表操作</h4>
                  <!-- 这里添加数据表操作的具体内容 -->
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 成功提示框 -->
    <div class="toast-container position-fixed top-0 start-50 translate-middle-x p-3">
      <div id="successToast" class="toast align-items-center text-bg-success border-0" role="alert">
        <div class="d-flex">
          <div class="toast-body">
            <i class="bi bi-check-circle me-2"></i>操作成功
          </div>
          <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Workbench'
}
</script>

<style scoped>
.workbench {
  background-color: #f8f9fa;
  min-height: calc(100vh - 56px);
}

.nav-tabs .nav-link {
  border: none;
  padding: 1rem;
  color: #6c757d;
}

.nav-tabs .nav-link.active {
  background-color: #fff;
  color: #0d6efd;
  border-bottom: 2px solid #0d6efd;
}

.card {
  border: none;
  box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
}
</style>