<template>
  <div class="home">
    <div class="hero-section">
      <div class="container py-5">
        <h1 class="text-center mb-4 text-white">AI 智能用例生成平台：大模型提效</h1>
        <div class="row align-items-center">
          <div class="col-md-6">
            <p class="mb-4 text-light">
              在快速迭代和高效研发的今天，<strong>AI 智能用例生成平台</strong>
              致力于结合大模型强大的自然语言处理能力，打造基于智能搜索和大模型的知识库解决方案。
              旨在帮助持续优化测试技术、提高测试效率，自动化地生成更全面和高质量的测试用例，打造 AI 时代的领先生产力。
            </p>
            <router-link to="/workbench" class="btn btn-danger btn-lg">开始体验</router-link>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Home'
}
</script>

<style scoped>
.home {
  min-height: 100vh;
  display: flex;
  align-items: center;
  background-image: url('/static/images/bg.png');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  position: relative;
}

.hero-section {
  width: 100%;
  position: relative;
  z-index: 1;
}

.hero-section::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(45deg, rgba(238, 73, 73, 0.1), rgba(0, 0, 0, 0.9));
  z-index: -1;
}

.btn-danger {
  padding: 0.75rem 2rem;
  font-size: 1.1rem;
  background-color: #ee4949;
  border: none;
  transition: all 0.3s ease;
}

.btn-danger:hover {
  background-color: #d63030;
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(238, 73, 73, 0.3);
}

.text-center {
  font-weight: 700;
}
</style>